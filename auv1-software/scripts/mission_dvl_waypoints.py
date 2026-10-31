"""Autonomous Waypoint Mission using DVL + 3D Dead Reckoning + LOS Guidance.

Runs identically in the 6-DOF simulator (`--link sim`) and on the Raspberry Pi 5
over the Pixhawk 6C (`--link real`), strictly following the Layering Rule
(Section 6 of the Context Brief).

Usage:
    # 1. Run in auv1-sim with DVL enabled (closed-loop speed + 3D DVL dead reckoning)
    python scripts/mission_dvl_waypoints.py --link sim --dvl

    # 2. Run in auv1-sim WITHOUT DVL (roadmap stub mode -> thrust-model dead reckoning)
    python scripts/mission_dvl_waypoints.py --link sim

    # 3. Run on real AUV1 hardware (Pi 5 -> BlueOS MAVLink + optional Water Linked DVL-A50)
    python scripts/mission_dvl_waypoints.py --link real --dvl --dvl-host 192.168.2.95

Safety & Failsafes:
- Never overrides ArduSub's leak or critical battery failsafes.
- Monitors battery voltage / leak status if exposed, plus enforces a DVL minimum
  altitude floor guard (`min_safe_altitude_m = 0.35 m`).
- Neutrals all fins (1500 us) and both ESCs (1500 us) and disarms on completion
  or Ctrl+C.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from auv1.controllers import PID
from auv1.dvl_io import DvlIO
from auv1.fin_mixer import deflection_to_pwm
from auv1.guidance import Waypoint, WaypointMissionRunner
from auv1.navigation import DeadReckoningNavigator
from auv1.vehicle_control import VehicleFlightController

FIN_OUTPUTS = {"top": 4, "bottom": 5, "left": 6, "right": 7}
STERN_OUTPUT = 8   # MAIN 8
BOW_OUTPUT = 9     # AUX 1

LOOP_HZ = 10.0
MAX_MISSION_S = 240.0

# Pool-safe 10 m x 10 m autonomous box + gate approach mission inside the
# 50 m x 25 m x 3 m simulator / test pool.
DEFAULT_WAYPOINTS = [
    Waypoint("submerge_north", north_m=8.0,  east_m=0.0,  depth_m=1.5, speed_mps=1.0, forward_effort=0.6),
    Waypoint("gate_east",      north_m=8.0,  east_m=8.0,  depth_m=1.5, speed_mps=1.0, forward_effort=0.6),
    Waypoint("south_leg",      north_m=0.0,  east_m=8.0,  depth_m=1.8, speed_mps=1.0, forward_effort=0.6,
             altitude_sp_m=1.2),  # holds 1.2 m above pool floor if DVL locked, else 1.8 m depth
    Waypoint("home_origin",    north_m=0.0,  east_m=0.0,  depth_m=1.5, speed_mps=0.9, forward_effort=0.55),
    Waypoint("surface_home",   north_m=0.0,  east_m=0.0,  depth_m=0.2, speed_mps=0.5, forward_effort=0.4,
             acceptance_radius_m=2.5),
]

TUNED_GAINS = {
    "depth_to_pitch": dict(kp=1.0, ki=0.05, kd=0.1, output_limit=1.0, integral_limit=0.3),
    "pitch":          dict(kp=0.04, ki=0.03, kd=0.004, output_limit=1.0, integral_limit=20.0),
    "roll":           dict(kp=0.02, ki=0.0, kd=0.002, output_limit=0.5, integral_limit=10.0),
    "heading":        dict(kp=0.02, ki=0.002, kd=0.005, output_limit=0.6, integral_limit=60.0),
}


def effort_to_pwm(effort: float) -> int:
    return int(1500 + max(-1.0, min(1.0, effort)) * 400)


def neutral_all(link) -> None:
    for out in FIN_OUTPUTS.values():
        link.set_servo_pwm(out, deflection_to_pwm(0.0))
    link.set_servo_pwm(STERN_OUTPUT, 1500)
    link.set_servo_pwm(BOW_OUTPUT, 1500)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--link", default="sim", choices=["sim", "real"])
    ap.add_argument("--connection", default="udpin:0.0.0.0:14551",
                    help="MAVLink endpoint when --link real")
    ap.add_argument("--dvl", action="store_true",
                    help="Enable DVL velocity fusion and closed-loop surge speed control")
    ap.add_argument("--dvl-host", default="192.168.2.95",
                    help="Water Linked DVL-A50 TCP IP address on --link real")
    ap.add_argument("--dvl-port", type=int, default=16171,
                    help="Water Linked DVL-A50 TCP port on --link real")
    ap.add_argument("--sim-path", default=None,
                    help="Path to auv1-sim repo (default ../auv1-sim)")
    args = ap.parse_args()

    if args.link == "sim":
        sim_dir = Path(args.sim_path) if args.sim_path else (REPO_ROOT.parent / "auv1-sim")
        if (sim_dir / "auv1sim" / "__init__.py").exists():
            sys.path.insert(0, str(sim_dir))
        from auv1sim.sim_link import SimLink
        link = SimLink(dvl_enabled=args.dvl)
        clock = link.now
        dvl_driver = None
        start_ne = link.pool_cfg.get("start_ned", [0.0, 0.0, 0.0])
    else:
        from auv1.mavlink_io import MavlinkIO
        link = MavlinkIO(args.connection)
        clock = time.monotonic
        dvl_driver = DvlIO(host=args.dvl_host, port=args.dvl_port, enabled=args.dvl)
        start_ne = [0.0, 0.0, 0.0]
        print(f"Waiting for Pixhawk heartbeat on {args.connection}...")
        if not link.wait_heartbeat():
            sys.exit("No heartbeat from Pixhawk 6C.")

    nav = DeadReckoningNavigator(dvl_enabled=args.dvl)
    nav.reset(north_m=float(start_ne[0]), east_m=float(start_ne[1]),
              depth_m=float(start_ne[2]))

    guidance = WaypointMissionRunner(DEFAULT_WAYPOINTS)
    ctl = VehicleFlightController()
    for loop_name, gains in TUNED_GAINS.items():
        setattr(ctl, loop_name, PID(**gains))

    link.arm()
    t0 = clock()
    last_stern_cmd = 0.0
    print(f"Starting autonomous waypoint mission (link={args.link}, dvl={args.dvl})")

    try:
        while not guidance.done:
            att = link.get_attitude()
            if att is None:
                continue
            roll_rad, pitch_rad, yaw_rad = att
            depth_m = link.get_depth()
            now = clock()

            # Read DVL + gyro + surface GPS from either SimLink or hardware I/O
            if args.link == "sim":
                dvl_pkt = link.get_dvl()
                gyro_rad_s = link.get_gyro()
                gps_ne = link.get_gps()
            else:
                reading = dvl_driver.poll(now=now) if dvl_driver else None
                dvl_pkt = reading.to_dict() if reading is not None else None
                gyro_rad_s = (0.0, 0.0, 0.0)
                gps_ne = None

            # 1) Navigation layer: 3D DVL dead-reckoning (with thrust fallback)
            nav_state = nav.update(
                now=now,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
                depth_m=depth_m,
                stern_effort=last_stern_cmd,
                gyro_rad_s=gyro_rad_s,
                dvl=dvl_pkt,
                gps_ne_m=gps_ne,
            )

            # 2) Guidance layer: LOS path following + underactuated speed schedule
            hdg_deg = math.degrees(yaw_rad) % 360.0
            g_out = guidance.update(
                now=now,
                north_m=nav_state["north_m"],
                east_m=nav_state["east_m"],
                depth_m=nav_state["depth_m"],
                heading_deg=hdg_deg,
                altitude_m=nav_state["altitude_m"],
            )

            # 3) Control layer: Pitch-and-drive depth cascade + speed-blended yaw
            #    + closed-loop surge speed PID when DVL is locked
            cmd = ctl.update(
                depth_m=nav_state["depth_m"],
                pitch_deg=math.degrees(pitch_rad),
                roll_deg=math.degrees(roll_rad),
                heading_deg=hdg_deg,
                depth_sp_m=g_out["depth_sp_m"],
                heading_sp_deg=g_out["heading_sp_deg"],
                forward=g_out["forward_effort"],
                u_mps=nav_state["u_m_s"] if nav_state["dvl_locked"] else None,
                speed_sp_mps=g_out["speed_sp_mps"] if nav_state["dvl_locked"] else None,
                now=now,
            )
            last_stern_cmd = cmd["stern"]

            # 4) Actuation layer: DO_SET_SERVO PWM outputs
            for fin_name, defl in cmd["fins"].items():
                link.set_servo_pwm(FIN_OUTPUTS[fin_name], deflection_to_pwm(defl))
            link.set_servo_pwm(STERN_OUTPUT, effort_to_pwm(cmd["stern"]))
            link.set_servo_pwm(BOW_OUTPUT, effort_to_pwm(cmd["bow"]))

            if (now - t0) > MAX_MISSION_S:
                print("\nMission timeout reached — surfacing and disarming.")
                break

            if args.link == "real":
                time.sleep(1.0 / LOOP_HZ)

        print("Mission complete.")
    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        neutral_all(link)
        link.disarm()
        if dvl_driver is not None:
            dvl_driver.close()


if __name__ == "__main__":
    main()
