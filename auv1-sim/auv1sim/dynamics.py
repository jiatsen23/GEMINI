"""Vehicle physics: actuator commands + water -> how the torpedo moves.

Frames (same as ArduPilot):
    WORLD  "NED": x north, y east, z DOWN. Depth is simply z. Origin = pool
           centre at the surface.
    BODY   x forward (nose), y starboard (right), z down (keel).
    Attitude = roll, pitch, yaw (radians). Pitch +ve = nose UP.
    Angular rates p, q, r about body x, y, z.

The model is a "Fossen-lite" 6-DOF rigid body:
    mass & added mass       - the hull plus the water it drags with it
    weight & buoyancy       - with a righting moment (CB above CG)
    quadratic drag          - per body axis, plus rotational damping
    stern thruster          - surge force at the tail, first-order lag
    bow tunnel thruster     - side force near the nose (yaw at low speed)
    four '+' fins           - lift from deflection AND from the flow angle
                              (the flow term is what makes a torpedo fly
                              straight - the fins weathercock it)
    pool walls & floor      - soft spring/damper contact at nose and tail
    free surface            - buoyancy fades out as the hull breaks surface

Deliberately NOT modelled (v1): hull lift / Munk moment, propeller torque,
currents, VBS (no VBS on the vehicle yet). Add them here when they matter.
"""

import math
import numpy as np

G = 9.81


# ── small rotation helpers ─────────────────────────────────────────────
def rotation_matrix(roll, pitch, yaw):
    """Body -> world rotation matrix R (world_vec = R @ body_vec)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def euler_rates(roll, pitch, pqr):
    """Convert body rates (p,q,r) to roll/pitch/yaw rates."""
    p, q, r = pqr
    cr, sr = math.cos(roll), math.sin(roll)
    cp, tp = math.cos(pitch), math.tan(pitch)
    return np.array([
        p + (q * sr + r * cr) * tp,
        q * cr - r * sr,
        (q * sr + r * cr) / cp,
    ])


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── the vehicle ────────────────────────────────────────────────────────
class Vehicle:
    def __init__(self, vehicle_cfg: dict, pool_cfg: dict):
        self.cfg = vehicle_cfg
        self.pool = pool_cfg
        h = vehicle_cfg["hull"]
        self.L = h["length_m"]
        self.r = h["diameter_m"] / 2.0
        self.m = h["mass_kg"]
        self.rho = pool_cfg["water_density_kg_m3"]
        self.W = self.m * G
        self.B = self.rho * G * h["displaced_volume_m3"]
        self.bg = h["bg_m"]

        # rigid inertia of a solid cylinder about its CG
        Ixx = 0.5 * self.m * self.r ** 2
        Iyy = self.m * (3 * self.r ** 2 + self.L ** 2) / 12.0
        am = vehicle_cfg["added_mass"]
        # total (rigid + added) mass per axis, kept diagonal for simplicity
        self.M_lin = np.array([self.m * (1 + am["surge"]),
                               self.m * (1 + am["sway"]),
                               self.m * (1 + am["heave"])])
        self.M_rot = np.array([Ixx * (1 + am["roll"]),
                               Iyy * (1 + am["pitch"]),
                               Iyy * (1 + am["yaw"])])

        d = vehicle_cfg["drag"]
        self.A_frontal = math.pi * self.r ** 2
        self.A_side = self.L * 2 * self.r
        self.drag_lin_coeff = np.array([0.5 * self.rho * d["cd_surge"] * self.A_frontal,
                                        0.5 * self.rho * d["cd_crossflow"] * self.A_side,
                                        0.5 * self.rho * d["cd_crossflow"] * self.A_side])
        self.rot_lin = np.array([d["roll_lin"], d["pitch_lin"], d["yaw_lin"]])
        self.rot_quad = np.array([d["roll_quad"], d["pitch_quad"], d["yaw_quad"]])

        self.stern = vehicle_cfg["stern_thruster"]
        self.bow = vehicle_cfg["bow_thruster"]
        self.fins = vehicle_cfg["fins"]
        self.fin_names = ["top", "bottom", "left", "right"]
        a = self.fins["radial_arm_m"]
        xf = self.fins["x_body_m"]
        # where each fin sits on the hull (body frame), and which body axis
        # its force acts along (vertical fins push sideways, horizontal
        # fins push up/down)
        self.fin_pos = {"top": np.array([xf, 0.0, -a]),
                        "bottom": np.array([xf, 0.0, +a]),
                        "left": np.array([xf, -a, 0.0]),
                        "right": np.array([xf, +a, 0.0])}
        self.fin_axis = {"top": 1, "bottom": 1, "left": 2, "right": 2}
        self.fin_max = math.radians(self.fins["max_deflection_deg"])
        self.fin_rate = math.radians(self.fins["servo_rate_deg_s"])

        self.reset()

    # ── state ──────────────────────────────────────────────────────────
    def reset(self):
        p = self.pool
        self.t = 0.0
        self.pos = np.array(p["start_ned"], dtype=float)           # NED, m
        self.att = np.array([0.0, 0.0, math.radians(p["start_heading_deg"])])
        self.v = np.zeros(3)                                       # body u,v,w
        self.w = np.zeros(3)                                       # body p,q,r
        # actuator states (what the hardware is actually doing right now)
        self.stern_thrust = 0.0                                    # N
        self.bow_thrust = 0.0                                      # N
        self.fin_defl = {n: 0.0 for n in self.fin_names}           # rad
        # commands (what the controller asked for)
        self.cmd_stern = 0.0
        self.cmd_bow = 0.0
        self.cmd_fin = {n: 0.0 for n in self.fin_names}            # -1..1
        self.armed = False
        self.wall_contact = False
        self.last_forces = {}
        self.acc_body = np.zeros(3)

    # ── commands from the control code ─────────────────────────────────
    def command_pwm(self, output: int, pwm_us: int):
        """Route a DO_SET_SERVO-style command to the right actuator.

        Same numbering as the Pixhawk: 4-7 fins, 8 stern ESC, 9 bow ESC.
        PWM 1100..1900 with 1500 = neutral -> effort -1..+1.
        """
        effort = max(-1.0, min(1.0, (pwm_us - 1500) / 400.0))
        if output == self.stern["servo_output"]:
            self.cmd_stern = effort
        elif output == self.bow["servo_output"]:
            self.cmd_bow = effort
        else:
            for name, out in self.fins["outputs"].items():
                if out == output:
                    self.cmd_fin[name] = effort
                    return
            raise ValueError(f"nothing is wired to output {output} in the sim")

    # ── one physics step ───────────────────────────────────────────────
    def step(self, dt: float):
        self._update_actuators(dt)
        F, T = self._forces_and_torques()
        self.last_forces = {"F": F.copy(), "T": T.copy()}

        # Newton-Euler in the body frame, with the w x (M v) "Coriolis" term
        dv = (F - np.cross(self.w, self.M_lin * self.v)) / self.M_lin
        dw = (T - np.cross(self.w, self.M_rot * self.w)) / self.M_rot
        # true acceleration of the CG in body axes (what an IMU would feel,
        # before gravity is added) - kept for the sensor model
        self.acc_body = dv + np.cross(self.w, self.v)
        self.v += dv * dt
        self.w += dw * dt

        R = rotation_matrix(*self.att)
        self.pos += (R @ self.v) * dt
        self.att += euler_rates(self.att[0], self.att[1], self.w) * dt
        self.att[0] = wrap_pi(self.att[0])
        self.att[2] = self.att[2] % (2 * math.pi)
        self.t += dt

    def _update_actuators(self, dt):
        if not self.armed:
            self.cmd_stern = 0.0
            self.cmd_bow = 0.0
        # thrusters: thrust ~ effort*|effort| (propeller law) with a lag
        e = self.cmd_stern
        target = (self.stern["max_fwd_N"] if e >= 0 else self.stern["max_rev_N"]) * e * abs(e)
        self.stern_thrust += (target - self.stern_thrust) * dt / self.stern["time_constant_s"]
        e = self.cmd_bow
        target = self.bow["max_N"] * e * abs(e)
        self.bow_thrust += (target - self.bow_thrust) * dt / self.bow["time_constant_s"]
        # fin servos: move toward the commanded angle at a limited rate
        for n in self.fin_names:
            target = self.cmd_fin[n] * self.fin_max
            delta = target - self.fin_defl[n]
            max_step = self.fin_rate * dt
            self.fin_defl[n] += max(-max_step, min(max_step, delta))

    def _forces_and_torques(self):
        F = np.zeros(3)
        T = np.zeros(3)
        R = rotation_matrix(*self.att)
        down_body = R.T @ np.array([0.0, 0.0, 1.0])   # "down" seen from the body
        u, v, w = self.v

        # weight (at CG = origin) and buoyancy (at CB, bg above the CG)
        submerged = min(1.0, max(0.0, (self.pos[2] + self.r) / (2 * self.r)))
        F += self.W * down_body
        Fb = -self.B * submerged * down_body
        F += Fb
        T += np.cross(np.array([0.0, 0.0, -self.bg]), Fb)

        # quadratic drag per body axis + rotational damping
        F += -self.drag_lin_coeff * np.abs(self.v) * self.v
        T += -(self.rot_lin * self.w + self.rot_quad * np.abs(self.w) * self.w)

        # stern thruster: along +x at the tail
        f = np.array([self.stern_thrust, 0.0, 0.0])
        F += f
        T += np.cross(np.array(self.stern["position_body_m"]), f)

        # bow tunnel thruster: along +y near the nose
        f = np.array([0.0, self.bow_thrust, 0.0])
        F += f
        T += np.cross(np.array(self.bow["position_body_m"]), f)

        # fins: lift = 0.5*rho*S*CLa * u * (sign*delta*u - flow_at_fin)
        k = 0.5 * self.rho * self.fins["area_each_m2"] * self.fins["cl_alpha_per_rad"]
        for n in self.fin_names:
            r_f = self.fin_pos[n]
            local_flow = self.v + np.cross(self.w, r_f)   # water velocity seen by the fin
            axis = self.fin_axis[n]
            sign = self.fins["force_sign"][n]
            lift = k * u * (sign * self.fin_defl[n] * u - local_flow[axis])
            # crude stall: never more than a 35 deg effective angle
            cap = k * u * u * math.radians(35.0)
            lift = max(-cap, min(cap, lift))
            f = np.zeros(3)
            f[axis] = lift
            F += f
            T += np.cross(r_f, f)

        # pool walls and floor: soft contact at nose and tail
        self.wall_contact = False
        for point_body in (np.array([self.L / 2, 0, 0]), np.array([-self.L / 2, 0, 0])):
            pw = self.pos + R @ point_body            # point in world frame
            vw = R @ (self.v + np.cross(self.w, point_body))
            f_world = self._contact_force(pw, vw)
            if np.any(f_world):
                self.wall_contact = True
                f_body = R.T @ f_world
                F += f_body
                T += np.cross(point_body, f_body)
        return F, T

    def _contact_force(self, p, vel):
        """Spring/damper push-back when a point pokes through a wall/floor."""
        k, c = self.pool["wall_stiffness_N_m"], self.pool["wall_damping_N_s_m"]
        half = np.array([self.pool["length_m"] / 2, self.pool["width_m"] / 2])
        f = np.zeros(3)
        for i in range(2):                          # x and y walls
            if p[i] > half[i]:
                f[i] = -k * (p[i] - half[i]) - c * max(0.0, vel[i])
            elif p[i] < -half[i]:
                f[i] = -k * (p[i] + half[i]) - c * min(0.0, vel[i])
        floor = self.pool["depth_m"]
        if p[2] > floor:                            # z down: below the floor
            f[2] = -k * (p[2] - floor) - c * max(0.0, vel[2])
        return f

    # ── convenience for logging ────────────────────────────────────────
    def truth(self) -> dict:
        return {
            "t": self.t,
            "pos": self.pos.tolist(),
            "att": self.att.tolist(),
            "v": self.v.tolist(),
            "w": self.w.tolist(),
            "stern_N": self.stern_thrust,
            "bow_N": self.bow_thrust,
            "fins_deg": {n: math.degrees(d) for n, d in self.fin_defl.items()},
            "wall": self.wall_contact,
        }
