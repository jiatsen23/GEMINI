"""Autonomous Waypoint Guidance & Underactuated Torpedo Behaviours.

Pure Python, no MAVLink imports (Layering Rule, Section 6).

Why underactuated guidance is special on AUV1:
- No vertical thrusters and no VBS yet: AUV1 changes and holds depth ONLY by
  pitching (+/-25 deg) with water flowing over the stern fins.
- If the vehicle stops moving forward while submerged, the fins lose lift and
  the slightly positive buoyancy / nose-up trim causes depth loss.
- At the same time, tight turns in a pool benefit from slowing down so the
  T200 bow tunnel thruster has yaw authority (`bow_authority` in
  vehicle_control.py) and turning radius stays small.

What this module provides:
1. Line-of-Sight (LOS) Path Following (`los_heading_deg`):
   Steers onto the line segment between `prev_wp` and `target_wp` ( eliminating
   cross-track drift from sideslip or currents) rather than naive point-to-point
   homing.
2. Underactuated Speed Scheduler (`schedule_speed`):
   - At large heading errors (> turn_threshold_deg), drops speed to a cornering
     speed where the bow thruster + fins work together, while enforcing a
     minimum flow floor (`min_submerged_mps`) whenever submerged.
3. Loiter / Orbit Station-Keeping (`orbit_guidance`):
   - Because AUV1 cannot hover at (N, E, D), "holding position" at a target or
     gate inspection point flies a gentle constant-depth circle of radius R
     around the point.
4. DVL Altitude-Above-Floor Hold & Floor Guard (`resolve_depth_setpoint`):
   - Converts an `altitude_sp_m` (height above pool floor using DVL range) into
     an equivalent `depth_sp_m` for `VehicleFlightController`, and clamps the
     target depth if DVL altitude breaches `min_safe_altitude_m`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from auv1.controllers import wrap_error_deg


@dataclass(frozen=True)
class Waypoint:
    name: str
    north_m: float
    east_m: float
    depth_m: float = 1.5
    speed_mps: float = 1.0           # target surge speed u (m/s) when DVL active
    forward_effort: float = 0.6      # fallback open-loop stern effort (0..1)
    acceptance_radius_m: float = 1.5
    depth_tolerance_m: float = 0.35
    altitude_sp_m: float | None = None  # optional DVL bottom-following height (m)
    loiter_s: float = 0.0            # if > 0, orbit waypoint for this many seconds


def distance_ne(n1: float, e1: float, n2: float, e2: float) -> float:
    return math.hypot(n2 - n1, e2 - e1)


def bearing_deg(n1: float, e1: float, n2: float, e2: float) -> float:
    """NED bearing in degrees [0, 360): 0 = North (+N), 90 = East (+E)."""
    return math.degrees(math.atan2(e2 - e1, n2 - n1)) % 360.0


def los_heading_deg(
    north_m: float,
    east_m: float,
    prev_n: float,
    prev_e: float,
    target_n: float,
    target_e: float,
    lookahead_m: float = 3.0,
) -> tuple[float, float, float]:
    """Fossen Line-of-Sight (LOS) path-following guidance law.

    Returns (heading_sp_deg, cross_track_error_m, along_track_remaining_m).
    """
    dn = target_n - prev_n
    de = target_e - prev_e
    seg_len = math.hypot(dn, de)
    if seg_len < 1e-3:
        # Degenerate segment: pure pursuit bearing straight to target
        return (
            bearing_deg(north_m, east_m, target_n, target_e),
            0.0,
            distance_ne(north_m, east_m, target_n, target_e),
        )

    path_angle = math.atan2(de, dn)
    cp, sp = math.cos(path_angle), math.sin(path_angle)

    # Project vehicle position into path-tangent / path-normal frame
    rn = north_m - prev_n
    re = east_m - prev_e
    s_along = rn * cp + re * sp
    e_cross = -rn * sp + re * cp   # +ve = vehicle is to starboard (right) of path

    # Steer back toward the path with lookahead distance Delta
    delta = max(0.5, lookahead_m)
    chi_d = path_angle - math.atan2(e_cross, delta)
    remaining = max(0.0, seg_len - s_along)
    return math.degrees(chi_d) % 360.0, e_cross, remaining


def orbit_guidance(
    north_m: float,
    east_m: float,
    center_n: float,
    center_e: float,
    radius_m: float = 3.0,
    clockwise: bool = True,
) -> float:
    """Compute heading setpoint (deg) to orbit a point without stopping."""
    dn = north_m - center_n
    de = east_m - center_e
    r = math.hypot(dn, de)
    psi_radial = math.atan2(de, dn)  # angle from center to vehicle
    direction = 1.0 if clockwise else -1.0
    # Tangential angle + proportional pull toward the orbit radius
    tangent = psi_radial + direction * (math.pi / 2.0)
    radial_err = r - max(1.0, radius_m)
    correction = direction * math.atan2(radial_err, max(1.0, radius_m))
    return math.degrees(tangent + correction) % 360.0


def resolve_depth_setpoint(
    target_depth_m: float,
    current_depth_m: float,
    altitude_m: float | None = None,
    altitude_sp_m: float | None = None,
    min_safe_altitude_m: float = 0.35,
) -> float:
    """Apply DVL bottom-altitude hold or floor-collision protection.

    Since water_column_depth = current_depth_m + altitude_m:
    - If `altitude_sp_m` is requested and DVL altitude is valid, tracks constant
      height above the bottom: depth_sp = (current_depth + altitude) - altitude_sp.
    - Enforces a hard floor ceiling so AUV1 never dives closer than
      `min_safe_altitude_m` to the pool floor.
    """
    if altitude_m is None or altitude_m <= 0.0:
        return max(0.0, target_depth_m)

    estimated_floor_depth = current_depth_m + altitude_m
    max_safe_depth = max(0.0, estimated_floor_depth - min_safe_altitude_m)

    if altitude_sp_m is not None:
        desired = estimated_floor_depth - altitude_sp_m
    else:
        desired = target_depth_m

    return max(0.0, min(max_safe_depth, desired))


class WaypointMissionRunner:
    """Stateful autonomous waypoint follower for AUV1."""

    def __init__(
        self,
        waypoints: list[Waypoint],
        *,
        lookahead_m: float = 3.5,
        turn_heading_err_deg: float = 35.0,
        turn_speed_mps: float = 0.45,
        turn_effort: float = 0.25,
        min_submerged_mps: float = 0.40,
        min_safe_altitude_m: float = 0.35,
    ):
        if not waypoints:
            raise ValueError("waypoints list must not be empty")
        self.waypoints = waypoints
        self.lookahead_m = lookahead_m
        self.turn_heading_err_deg = turn_heading_err_deg
        self.turn_speed_mps = turn_speed_mps
        self.turn_effort = turn_effort
        self.min_submerged_mps = min_submerged_mps
        self.min_safe_altitude_m = min_safe_altitude_m

        self.index = 0
        self.prev_ne: tuple[float, float] | None = None
        self._loiter_start_t: float | None = None

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def current_waypoint(self) -> Waypoint | None:
        if self.done:
            return None
        return self.waypoints[self.index]

    def update(
        self,
        *,
        now: float,
        north_m: float,
        east_m: float,
        depth_m: float,
        heading_deg: float,
        altitude_m: float | None = None,
    ) -> dict:
        """Compute guidance targets for the current tick."""
        if self.done:
            return {
                "done": True,
                "wp_name": "complete",
                "depth_sp_m": 0.2,
                "heading_sp_deg": heading_deg,
                "speed_sp_mps": 0.0,
                "forward_effort": 0.0,
                "dist_to_wp_m": 0.0,
                "cross_track_m": 0.0,
            }

        wp = self.waypoints[self.index]
        if self.prev_ne is None:
            self.prev_ne = (north_m, east_m)

        dist_ne = distance_ne(north_m, east_m, wp.north_m, wp.east_m)
        depth_sp = resolve_depth_setpoint(
            wp.depth_m,
            depth_m,
            altitude_m=altitude_m,
            altitude_sp_m=wp.altitude_sp_m,
            min_safe_altitude_m=self.min_safe_altitude_m,
        )
        depth_err = abs(depth_sp - depth_m)

        # Check waypoint arrival / loiter transitions
        arrived_horizontally = dist_ne <= wp.acceptance_radius_m
        arrived_vertically = depth_err <= wp.depth_tolerance_m

        if arrived_horizontally and arrived_vertically:
            if wp.loiter_s > 0.0:
                if self._loiter_start_t is None:
                    self._loiter_start_t = now
                if (now - self._loiter_start_t) >= wp.loiter_s:
                    self._advance(wp.north_m, wp.east_m)
            else:
                self._advance(wp.north_m, wp.east_m)

            if self.done:
                return {
                    "done": True,
                    "wp_name": "complete",
                    "depth_sp_m": 0.2,
                    "heading_sp_deg": heading_deg,
                    "speed_sp_mps": 0.0,
                    "forward_effort": 0.0,
                    "dist_to_wp_m": 0.0,
                    "cross_track_m": 0.0,
                }
            wp = self.waypoints[self.index]
            dist_ne = distance_ne(north_m, east_m, wp.north_m, wp.east_m)
            depth_sp = resolve_depth_setpoint(
                wp.depth_m,
                depth_m,
                altitude_m=altitude_m,
                altitude_sp_m=wp.altitude_sp_m,
                min_safe_altitude_m=self.min_safe_altitude_m,
            )

        # Compute heading setpoint: Orbit if loitering, else LOS path following
        if self._loiter_start_t is not None:
            hdg_sp = orbit_guidance(
                north_m, east_m, wp.north_m, wp.east_m,
                radius_m=max(2.0, wp.acceptance_radius_m),
            )
            xte = dist_ne - max(2.0, wp.acceptance_radius_m)
        else:
            hdg_sp, xte, _ = los_heading_deg(
                north_m, east_m,
                self.prev_ne[0], self.prev_ne[1],
                wp.north_m, wp.east_m,
                lookahead_m=self.lookahead_m,
            )

        # Underactuated speed scheduling:
        # Slow down for sharp turns so the bow thruster has authority, but
        # never drop below min_submerged_mps when holding depth underwater.
        hdg_err = abs(wrap_error_deg(hdg_sp, heading_deg))
        if hdg_err > self.turn_heading_err_deg and wp.speed_mps > 0.0:
            speed_sp = self.turn_speed_mps
            fwd_eff = self.turn_effort
        else:
            speed_sp = wp.speed_mps
            fwd_eff = wp.forward_effort

        if depth_sp > 0.35 and speed_sp > 0.0:
            speed_sp = max(self.min_submerged_mps, speed_sp)

        return {
            "done": False,
            "wp_name": wp.name,
            "depth_sp_m": depth_sp,
            "heading_sp_deg": hdg_sp,
            "speed_sp_mps": speed_sp,
            "forward_effort": fwd_eff,
            "dist_to_wp_m": dist_ne,
            "cross_track_m": xte,
        }

    def _advance(self, reached_n: float, reached_e: float) -> None:
        self.prev_ne = (reached_n, reached_e)
        self.index += 1
        self._loiter_start_t = None
