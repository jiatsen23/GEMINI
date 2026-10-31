"""Onboard 3D Dead-Reckoning & DVL-Fused Navigation Estimator.

Pure Python, no MAVLink or socket imports (Layering Rule, Section 6).

Why AUV1 needs 3D body-velocity dead reckoning:
1. Underactuated pitch-and-drive: AUV1 pitches up to +/-25 deg to change
   depth. A 2D estimator that assumes horizontal speed == surge speed
   over-estimates North/East distance by 1/cos(25 deg) (~10%) during dives.
2. Fin & bow-thruster sideslip: When turning, the torpedo develops non-zero
   sway velocity (v_y) and yaw rate (r). Without a DVL, pure heading-walk
   dead reckoning ignores sideslip.
3. Lever-arm compensation: Because AUV1 is 1.5 m long and pitches/yaws
   actively, a DVL mounted at r_dvl = [x, y, z] from the CG measures:
       v_dvl = v_cg + (omega x r_dvl)
   We subtract (omega x r_dvl) before rotating into the NED world frame.
4. Graceful fallback (Feature Flag / Bottom-Lock Loss):
   - When the DVL is enabled, has bottom lock (`valid=True`), and its Figure
     of Merit (`fom_m_s`) is within threshold, we integrate lever-arm
     corrected 3D DVL velocity.
   - When the DVL is disabled (stub mode today) or loses bottom lock (e.g.
     steep pitch near pool wall, or below minimum blanking altitude), we
     fall back smoothly to a lagged propeller-thrust surge model so the
     mission never loses a continuous state estimate.
"""

from __future__ import annotations

import math

# Default hydrodynamic fallback parameters (matches auv1-sim vehicle.json)
DEFAULT_SPEED_PER_EFFORT = 2.3    # m/s steady-state surge at full stern effort
DEFAULT_THRUSTER_TAU_S = 0.15     # first-order thruster + hull response lag (s)
DEFAULT_MAX_FOM_M_S = 0.08        # reject DVL frames with uncertainty > 8 cm/s
DEFAULT_MIN_ALTITUDE_M = 0.05     # Water Linked A50 minimum blanking distance
DEFAULT_STALE_DVL_S = 0.5         # fallback to thrust model if DVL older than 0.5 s


def rotate_body_to_ned(u: float, v: float, w: float,
                       roll_rad: float, pitch_rad: float,
                       yaw_rad: float) -> tuple[float, float, float]:
    """Rotate a 3D vector from body frame (Fwd, Stbd, Down) to world (N, E, D)."""
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

    vn = (cy * cp) * u + (cy * sp * sr - sy * cr) * v + (cy * sp * cr + sy * sr) * w
    ve = (sy * cp) * u + (sy * sp * sr + cy * cr) * v + (sy * sp * cr - cy * sr) * w
    vd = (-sp) * u + (cp * sr) * v + (cp * cr) * w
    return vn, ve, vd


def lever_arm_correct(vx: float, vy: float, vz: float,
                      p_rad_s: float, q_rad_s: float, r_rad_s: float,
                      r_dvl_body_m: tuple[float, float, float]) -> tuple[float, float, float]:
    """Subtract omega x r_dvl from DVL-measured velocity to get CG velocity.

    v_cg = v_dvl - (w x r_dvl)
    """
    rx, ry, rz = r_dvl_body_m
    wx_r_x = q_rad_s * rz - r_rad_s * ry
    wx_r_y = r_rad_s * rx - p_rad_s * rz
    wx_r_z = p_rad_s * ry - q_rad_s * rx
    return vx - wx_r_x, vy - wx_r_y, vz - wx_r_z


class DeadReckoningNavigator:
    """Unified onboard dead-reckoning estimator with DVL + thrust-model fallback."""

    def __init__(
        self,
        *,
        dvl_enabled: bool = True,
        r_dvl_body_m: tuple[float, float, float] = (0.20, 0.0, 0.09),
        speed_per_effort: float = DEFAULT_SPEED_PER_EFFORT,
        thruster_tau_s: float = DEFAULT_THRUSTER_TAU_S,
        max_fom_m_s: float = DEFAULT_MAX_FOM_M_S,
        min_altitude_m: float = DEFAULT_MIN_ALTITUDE_M,
        stale_dvl_s: float = DEFAULT_STALE_DVL_S,
    ):
        self.dvl_enabled = dvl_enabled
        self.r_dvl_body_m = r_dvl_body_m
        self.speed_per_effort = speed_per_effort
        self.thruster_tau_s = thruster_tau_s
        self.max_fom_m_s = max_fom_m_s
        self.min_altitude_m = min_altitude_m
        self.stale_dvl_s = stale_dvl_s
        self.reset()

    def reset(self, north_m: float = 0.0, east_m: float = 0.0,
              depth_m: float = 0.0) -> None:
        self.north_m = north_m
        self.east_m = east_m
        self.depth_m = depth_m
        self.v_body = (0.0, 0.0, 0.0)       # (u, v, w) in m/s at CG
        self.v_ned = (0.0, 0.0, 0.0)        # (vn, ve, vd) in m/s
        self.altitude_m: float | None = None
        self.dvl_locked = False
        self.has_gps_fix = False
        self.pos_uncertainty_m = 0.0        # grows with dead-reckoning drift
        self._u_model = 0.0
        self._last_t: float | None = None

    def update(
        self,
        *,
        now: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
        depth_m: float,
        stern_effort: float = 0.0,
        gyro_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dvl: dict | None = None,
        gps_ne_m: tuple[float, float] | list[float] | None = None,
    ) -> dict:
        """Advance state estimate by one tick and return the navigation state."""
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now

        # Always keep the lagged propeller-model surge speed warm so fallback
        # transitions are bumpless if a DVL beam drops out mid-manoeuvre.
        target_u = self.speed_per_effort * stern_effort * abs(stern_effort)
        if dt > 0 and self.thruster_tau_s > 0:
            alpha = min(1.0, dt / self.thruster_tau_s)
            self._u_model += alpha * (target_u - self._u_model)
        else:
            self._u_model = target_u

        # Check whether we have a fresh, valid DVL bottom-lock measurement
        use_dvl = False
        if self.dvl_enabled and dvl is not None:
            dvl_age = now - float(dvl.get("t", now))
            valid = bool(dvl.get("valid", False))
            fom = float(dvl.get("fom_m_s", 9.99))
            alt = float(dvl.get("altitude_m", -1.0))
            if (dvl_age <= self.stale_dvl_s
                    and valid
                    and fom <= self.max_fom_m_s
                    and alt >= self.min_altitude_m):
                use_dvl = True
                self.altitude_m = alt
            elif alt >= self.min_altitude_m:
                self.altitude_m = alt
            else:
                self.altitude_m = None
        else:
            self.altitude_m = None

        if use_dvl and dvl is not None:
            p, q, r = gyro_rad_s
            u, v, w = lever_arm_correct(
                float(dvl["vx_m_s"]),
                float(dvl["vy_m_s"]),
                float(dvl["vz_m_s"]),
                p, q, r,
                self.r_dvl_body_m,
            )
            self.v_body = (u, v, w)
            self._u_model = u   # sync fallback state to measured surge speed
            self.dvl_locked = True
            drift_rate_m_s = float(dvl.get("fom_m_s", 0.01))
        else:
            # Fallback: lagged surge model along body +x, zero sway/heave
            self.v_body = (self._u_model, 0.0, 0.0)
            self.dvl_locked = False
            drift_rate_m_s = 0.15 * abs(self._u_model) + 0.02

        # Rotate 3D body velocity into NED world frame using full 3D attitude
        vn, ve, vd = rotate_body_to_ned(
            self.v_body[0], self.v_body[1], self.v_body[2],
            roll_rad, pitch_rad, yaw_rad,
        )
        self.v_ned = (vn, ve, vd)

        # Integrate horizontal position (North, East)
        self.north_m += vn * dt
        self.east_m += ve * dt
        self.pos_uncertainty_m += drift_rate_m_s * dt

        # Depth comes directly from the Bar30 pressure sensor (1 cm accuracy,
        # zero cumulative drift), optionally blended with vertical velocity
        self.depth_m = depth_m

        # Surface GPS fix correction (only available when antenna < 0.1 m depth)
        if gps_ne_m is not None:
            gn, ge = float(gps_ne_m[0]), float(gps_ne_m[1])
            if not self.has_gps_fix:
                self.north_m, self.east_m = gn, ge
                self.has_gps_fix = True
            else:
                self.north_m += 0.3 * (gn - self.north_m)
                self.east_m += 0.3 * (ge - self.east_m)
            self.pos_uncertainty_m = 1.5

        return self.state()

    def state(self) -> dict:
        return {
            "north_m": self.north_m,
            "east_m": self.east_m,
            "depth_m": self.depth_m,
            "u_m_s": self.v_body[0],
            "v_m_s": self.v_body[1],
            "w_m_s": self.v_body[2],
            "vn_m_s": self.v_ned[0],
            "ve_m_s": self.v_ned[1],
            "vd_m_s": self.v_ned[2],
            "altitude_m": self.altitude_m,
            "dvl_locked": self.dvl_locked,
            "pos_uncertainty_m": self.pos_uncertainty_m,
        }
