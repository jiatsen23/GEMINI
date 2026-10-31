"""State estimator: noisy readings in, best-guess vehicle state out.

On the real vehicle this job is done either by ArduSub's EKF or by our
onboard `auv1.navigation.DeadReckoningNavigator` on the Pi 5:

    attitude   complementary filter - integrate the gyro (fast, drifts),
               nudge toward the accelerometer's gravity direction (slow,
               doesn't drift) for roll/pitch and toward the compass for yaw
    depth      low-pass of the pressure depth
    position   GPS when at the surface; underwater, if a DVL reading is
               present and valid (`reading["dvl"]["valid"]`), rotate the
               lever-arm-compensated 3D body velocity into NED and integrate.
               Otherwise fall back to thrust-effort dead reckoning.
"""

import math
import numpy as np
from .dynamics import rotation_matrix, wrap_pi

ACCEL_WEIGHT = 0.02      # how hard the accelerometer pulls roll/pitch per tick
MAG_WEIGHT = 0.05        # how hard the compass pulls yaw per tick
DEPTH_ALPHA = 0.5        # depth low-pass (1 = no filtering)
SPEED_PER_EFFORT = 2.3   # m/s of forward speed per unit stern effort
                         # (the physics says ~2.5 - the estimator is a
                         # little wrong on purpose, like every real model)


class Estimator:
    def __init__(self, r_dvl_body_m=(0.2, 0.0, 0.09)):
        self.att = np.zeros(3)          # roll, pitch, yaw (rad)
        self.depth = 0.0
        self.pos_ne = np.zeros(2)
        self.v_body = np.zeros(3)       # estimated (u, v, w) m/s at CG
        self.altitude_m = None
        self.dvl_locked = False
        self.has_fix = False
        self.r_dvl = np.array(r_dvl_body_m, dtype=float)
        self._t = None

    def update(self, reading: dict, stern_effort: float):
        t = reading["t"]
        dt = 0.0 if self._t is None else t - self._t
        self._t = t

        # 1) gyro integration (body rates -> approximate Euler rates)
        g = np.array(reading["gyro_rad_s"])
        self.att += g * dt

        # 2) accelerometer says which way is down -> roll & pitch correction
        ax, ay, az = reading["accel_m_s2"]
        roll_acc = math.atan2(-ay, -az)
        pitch_acc = math.atan2(ax, math.hypot(ay, az))
        self.att[0] += ACCEL_WEIGHT * wrap_pi(roll_acc - self.att[0])
        self.att[1] += ACCEL_WEIGHT * wrap_pi(pitch_acc - self.att[1])

        # 3) compass -> yaw correction
        yaw_mag = math.radians(reading["mag_heading_deg"])
        self.att[2] += MAG_WEIGHT * wrap_pi(yaw_mag - self.att[2])
        self.att[2] %= 2 * math.pi

        # 4) depth
        self.depth += DEPTH_ALPHA * (reading["depth_m"] - self.depth)

        # 5) velocity & position: GPS at surface, else DVL 3D dead reckoning,
        #    else fallback thrust-effort dead reckoning
        dvl = reading.get("dvl")
        if dvl is not None and dvl.get("valid", False):
            v_dvl = np.array([dvl["vx_m_s"], dvl["vy_m_s"], dvl["vz_m_s"]])
            # Lever-arm compensation: v_cg = v_dvl - (w x r_dvl)
            self.v_body = v_dvl - np.cross(g, self.r_dvl)
            self.altitude_m = dvl.get("altitude_m")
            self.dvl_locked = True
            R = rotation_matrix(self.att[0], self.att[1], self.att[2])
            v_ned = R @ self.v_body
            self.pos_ne += v_ned[:2] * dt
        else:
            speed = SPEED_PER_EFFORT * stern_effort
            self.v_body = np.array([speed, 0.0, 0.0])
            self.altitude_m = None
            self.dvl_locked = False
            yaw = self.att[2]
            self.pos_ne += speed * dt * np.array([math.cos(yaw), math.sin(yaw)])

        if reading["gps_ne_m"] is not None:
            gps = np.array(reading["gps_ne_m"])
            if not self.has_fix:
                self.pos_ne = gps
                self.has_fix = True
            else:
                self.pos_ne += 0.3 * (gps - self.pos_ne)

    def state(self) -> dict:
        return {
            "att": self.att.tolist(),
            "depth": self.depth,
            "pos_ne": self.pos_ne.tolist(),
            "v_body": self.v_body.tolist(),
            "altitude_m": self.altitude_m,
            "dvl_locked": self.dvl_locked,
            "has_fix": self.has_fix,
        }
