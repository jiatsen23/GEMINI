"""Sensor models: the TRUE state goes in, noisy readings come out.

Every sensor on the vehicle is a function of the true state plus noise and
bias. This is the honest part of a simulator - a controller that only
ever saw perfect numbers falls over the first time it meets a real Bar30.

    depth   pressure sensor  -> true depth + fixed bias + white noise
    gyro    IMU rates        -> true p,q,r + fixed bias + white noise
    accel   IMU              -> specific force (gravity included) + noise
    mag     compass heading  -> true yaw + local offset + white noise
    gps     surface only     -> N,E position + noise, and ONLY when the
                                antenna is within gps_max_depth_m of the
                                surface. Underwater there is no GPS. Full stop.
    dvl     4-beam Janus DVL -> 3D body velocity at transducer (v_cg + w x r_dvl)
                                + scale factor + white noise + bottom altitude,
                                gated by feature flag `dvl_enabled`, tilt cone,
                                and minimum blanking altitude.
    sonar   Ping360 / multibeam -> 2D horizontal acoustic raycast against pool
                                walls and submerged targets (gate posts, buoys)
                                with range noise and reflectivity intensity.
"""

import math
import numpy as np
from .dynamics import rotation_matrix, G


def _ray_circle_intersect(ox: float, oy: float, dx: float, dy: float,
                          cx: float, cy: float, r: float) -> float | None:
    """Return smallest positive ray distance to a circle at (cx, cy) of radius r."""
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t1 = (-b - sq) * 0.5
    t2 = (-b + sq) * 0.5
    if t1 > 0.2:
        return t1
    if t2 > 0.2:
        return t2
    return None


def _ray_pool_wall_distance(ox: float, oy: float, dx: float, dy: float,
                            half_L: float, half_W: float) -> float:
    """Return distance along unit ray (dx, dy) from (ox, oy) to pool walls."""
    best = 999.0
    if abs(dx) > 1e-6:
        for wx in (-half_L, half_L):
            t = (wx - ox) / dx
            if 0.05 < t < best:
                y_hit = oy + t * dy
                if -half_W <= y_hit <= half_W:
                    best = t
    if abs(dy) > 1e-6:
        for wy in (-half_W, half_W):
            t = (wy - oy) / dy
            if 0.05 < t < best:
                x_hit = ox + t * dx
                if -half_L <= x_hit <= half_L:
                    best = t
    return best


class Sensors:
    def __init__(self, sensor_cfg: dict, seed: int = 1):
        self.cfg = sensor_cfg
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = np.array(sensor_cfg["gyro_bias_rad_s"])
        self._last_gps_t = -1e9

    def read(self, vehicle) -> dict:
        c = self.cfg
        rng = self.rng
        roll, pitch, yaw = vehicle.att

        depth = vehicle.pos[2] + c["depth_bias_m"] + rng.normal(0, c["depth_noise_m"])

        gyro = vehicle.w + self.gyro_bias + rng.normal(0, c["gyro_noise_rad_s"], 3)

        R = rotation_matrix(roll, pitch, yaw)
        gravity_body = R.T @ np.array([0.0, 0.0, G])
        accel = vehicle.acc_body - gravity_body + rng.normal(0, c["accel_noise_m_s2"], 3)

        heading = math.degrees(yaw) + c["mag_heading_bias_deg"] \
            + rng.normal(0, c["mag_heading_noise_deg"])
        heading %= 360.0

        gps = None
        if (vehicle.pos[2] <= c["gps_max_depth_m"]
                and vehicle.t - self._last_gps_t >= 1.0 / c["gps_hz"]):
            self._last_gps_t = vehicle.t
            gps = (vehicle.pos[:2] + rng.normal(0, c["gps_noise_m"], 2)).tolist()

        dvl = None
        if c.get("dvl_enabled", False):
            r_dvl = np.array(c.get("dvl_position_body_m", [0.2, 0.0, 0.09]))
            v_dvl_true = vehicle.v + np.cross(vehicle.w, r_dvl)
            scale = 1.0 + c.get("dvl_scale_bias", 0.003)
            vel_noise = c.get("dvl_vel_noise_m_s", 0.005)
            v_meas = scale * v_dvl_true + rng.normal(0, vel_noise, 3)

            true_vertical_alt = max(0.0, vehicle.pool["depth_m"] - vehicle.pos[2] - r_dvl[2])
            tilt_cos = max(0.1, math.cos(roll) * math.cos(pitch))
            tilt_deg = math.degrees(math.acos(min(1.0, max(-1.0, math.cos(roll) * math.cos(pitch)))))
            alt_meas = true_vertical_alt + rng.normal(0, c.get("dvl_alt_noise_m", 0.01))

            min_alt = c.get("dvl_min_alt_m", 0.05)
            max_tilt = c.get("dvl_max_tilt_deg", 35.0)
            valid = (true_vertical_alt >= min_alt) and (tilt_deg <= max_tilt) and (vehicle.pos[2] >= 0.05)
            fom = vel_noise / tilt_cos if valid else 0.99

            dvl = {
                "t": vehicle.t,
                "vx_m_s": float(v_meas[0]),
                "vy_m_s": float(v_meas[1]),
                "vz_m_s": float(v_meas[2]),
                "altitude_m": float(max(0.0, alt_meas)) if valid else -1.0,
                "fom_m_s": float(fom),
                "valid": bool(valid),
            }

        sonar = None
        if c.get("sonar_enabled", False):
            max_r = float(c.get("sonar_max_range_m", 18.0))
            n_beams = int(c.get("sonar_beams_per_tick", 16))
            r_noise = float(c.get("sonar_range_noise_m", 0.04))
            half_L = vehicle.pool["length_m"] / 2.0
            half_W = vehicle.pool["width_m"] / 2.0
            targets = vehicle.pool.get("targets", [])
            ox, oy = float(vehicle.pos[0]), float(vehicle.pos[1])

            beams = []
            for k in range(n_beams):
                ang_body = -math.pi + (2.0 * math.pi * k) / n_beams
                beam_yaw = yaw + ang_body
                dx, dy = math.cos(beam_yaw), math.sin(beam_yaw)

                best_dist = _ray_pool_wall_distance(ox, oy, dx, dy, half_L, half_W)
                best_inten = 0.78

                for tgt in targets:
                    td = _ray_circle_intersect(
                        ox, oy, dx, dy,
                        float(tgt["north_m"]), float(tgt["east_m"]),
                        float(tgt.get("radius_m", 0.2)),
                    )
                    if td is not None and td < best_dist:
                        best_dist = td
                        best_inten = float(tgt.get("reflectivity", 0.95))

                if best_dist <= max_r:
                    r_meas = max(0.2, best_dist + float(rng.normal(0.0, r_noise)))
                    beams.append({
                        "angle_body_rad": ang_body,
                        "range_m": r_meas,
                        "max_range_m": max_r,
                        "intensity": best_inten,
                        "hit": True,
                    })
                else:
                    beams.append({
                        "angle_body_rad": ang_body,
                        "range_m": max_r,
                        "max_range_m": max_r,
                        "intensity": 0.0,
                        "hit": False,
                    })

            sonar = {"t": vehicle.t, "beams": beams}

        return {
            "t": vehicle.t,
            "depth_m": depth,
            "gyro_rad_s": gyro.tolist(),
            "accel_m_s2": accel.tolist(),
            "mag_heading_deg": heading,
            "gps_ne_m": gps,
            "dvl": dvl,
            "sonar": sonar,
        }
