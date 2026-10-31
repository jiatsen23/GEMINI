"""Autonomous Natural-Language + DVL/Sonar Fusion SLAM + Distilled AI Runner.

Ties together the entire AUV1 AI & Sensor-Fusion Stack on the Pi 5 or in
`auv1-sim`:
1. NaturalLanguagePlanner (`auv1/nl_planner.py`):
   Translates natural language ("map out the pool", "find and pass through the
   gate", "inspect the perimeter walls") into executable Waypoint missions.
2. DvlSonarFusionSLAM (`auv1/sensor_fusion.py`):
   Fuses DVL 3D velocities + IMU/Bar30 + Scanning Sonar into a drift-corrected
   pose estimate, a 2D Log-Odds Occupancy Grid, and camera-like Semantic
   Detections (`gate`, `buoy`).
3. Teacher-Student Distilled Control (`auv1/llm_distill.py`):
   Trains a fast 10->16->3 neural policy online from the AI Teacher's sonar
   obstacle avoidance & sideslip corrections, feeding into
   `VehicleFlightController` (`auv1/vehicle_control.py`).

Usage:
    python scripts/mission_ai_sonar.py --prompt "map out the pool" --link sim --dvl --sonar
    python scripts/mission_ai_sonar.py --prompt "find and pass through the gate" --link sim --dvl --sonar
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

from auv1.controllers import PID, wrap_error_deg
from auv1.dvl_io import DvlIO
from auv1.fin_mixer import deflection_to_pwm
from auv1.guidance import WaypointMissionRunner
from auv1.llm_distill import (
    DistilledStudentPolicy,
    LLMTeacherSupervisor,
    build_feature_vector,
)
from auv1.nl_planner import NaturalLanguagePlanner, generate_gate_search_and_pass
from auv1.sensor_fusion import DvlSonarFusionSLAM
from auv1.sonar_io import SonarIO
from auv1.vehicle_control import VehicleFlightController

FIN_OUTPUTS = {"top": 4, "bottom": 5, "left": 6, "right": 7}
STERN_OUTPUT = 8
BOW_OUTPUT = 9

LOOP_HZ = 10.0
MAX_MISSION_S = 300.0

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
    ap.add_argument("--prompt", default="I want to map out the pool",
                    help="Natural language mission instruction")
    ap.add_argument("--link", default="sim", choices=["sim", "real"])
    ap.add_argument("--connection", default="udpin:0.0.0.0:14551")
    ap.add_argument("--dvl", action="store_true", default=True)
    ap.add_argument("--sonar", action="store_true", default=True)
    ap.add_argument("--distill", action="store_true", default=True,
                    help="Enable online Teacher->Student AI control distillation")
    ap.add_argument("--sim-path", default=None)
    args = ap.parse_args()

    if args.link == "sim":
        sim_dir = Path(args.sim_path) if args.sim_path else (REPO_ROOT.parent / "auv1-sim")
        if (sim_dir / "auv1sim" / "__init__.py").exists():
            sys.path.insert(0, str(sim_dir))
        from auv1sim.sim_link import SimLink
        link = SimLink(dvl_enabled=args.dvl, sonar_enabled=args.sonar)
        clock = link.now
        dvl_io = sonar_io = None
        start_ne = link.pool_cfg.get("start_ned", [0.0, 0.0, 0.0])
        pool_L = float(link.pool_cfg.get("length_m", 50.0))
        pool_W = float(link.pool_cfg.get("width_m", 25.0))
    else:
        from auv1.mavlink_io import MavlinkIO
        link = MavlinkIO(args.connection)
        clock = time.monotonic
        dvl_io = DvlIO(enabled=args.dvl)
        sonar_io = SonarIO(enabled=args.sonar)
        start_ne = [0.0, 0.0, 0.0]
        pool_L, pool_W = 50.0, 25.0
        if not link.wait_heartbeat():
            sys.exit("No heartbeat from Pixhawk 6C.")

    slam = DvlSonarFusionSLAM(
        pool_length_m=pool_L,
        pool_width_m=pool_W,
        dvl_enabled=args.dvl,
        sonar_enabled=args.sonar,
    )
    slam.reset(north_m=float(start_ne[0]), east_m=float(start_ne[1]),
               depth_m=float(start_ne[2]))

    planner = NaturalLanguagePlanner(pool_length_m=pool_L, pool_width_m=pool_W)
    plan = planner.plan_from_text(args.prompt, slam.state())
    print(f"[NL Planner] Intent: {plan['intent']} -> {plan['explanation']}")
    guidance = WaypointMissionRunner(plan["waypoints"])

    teacher = LLMTeacherSupervisor()
    student = DistilledStudentPolicy()
    ctl = VehicleFlightController()
    for loop_name, gains in TUNED_GAINS.items():
        setattr(ctl, loop_name, PID(**gains))

    link.arm()
    t0 = clock()
    last_stern = 0.0
    gate_replanned = False

    try:
        while not guidance.done:
            att = link.get_attitude()
            if att is None:
                continue
            roll_rad, pitch_rad, yaw_rad = att
            depth_m = link.get_depth()
            now = clock()

            if args.link == "sim":
                dvl_pkt = link.get_dvl()
                sonar_pkt = link.get_sonar()
                gyro = link.get_gyro()
                gps = link.get_gps()
            else:
                dr = dvl_io.poll(now=now) if dvl_io else None
                sr = sonar_io.poll(now=now) if sonar_io else None
                dvl_pkt = dr.to_dict() if dr else None
                sonar_pkt = sr.to_dict() if sr else None
                gyro = (0.0, 0.0, 0.0)
                gps = None

            # 1) DVL + Sonar SLAM Fusion + Semantic Detection
            state = slam.update(
                now=now,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
                depth_m=depth_m,
                stern_effort=last_stern,
                gyro_rad_s=gyro,
                dvl=dvl_pkt,
                sonar_scan=sonar_pkt,
                gps_ne_m=gps,
            )

            # Dynamic Replanning: if mission is "gate_mission" and Sonar+DVL
            # just discovered the gate posts, lock onto the gate automatically!
            if plan["intent"] == "gate_mission" and not gate_replanned:
                gate_det = next((d for d in state["detections"] if d["label"] == "gate"), None)
                if gate_det is not None and gate_det["confidence"] >= 0.75:
                    gate_replanned = True
                    new_wps = generate_gate_search_and_pass(known_gate=gate_det)
                    guidance = WaypointMissionRunner(new_wps)
                    print(f"[Acoustic Vision] Gate locked at N={gate_det['north_m']}m, "
                          f"E={gate_det['east_m']}m (conf={gate_det['confidence']*100:.0f}%)! Replanning.")

            # 2) LOS Guidance
            hdg_deg = math.degrees(yaw_rad) % 360.0
            g_out = guidance.update(
                now=now,
                north_m=state["north_m"],
                east_m=state["east_m"],
                depth_m=state["depth_m"],
                heading_deg=hdg_deg,
                altitude_m=state["altitude_m"],
            )

            # 3) Teacher-Student AI Control Distillation
            depth_err = g_out["depth_sp_m"] - state["depth_m"]
            hdg_err = wrap_error_deg(g_out["heading_sp_deg"], hdg_deg)
            feats = build_feature_vector(
                depth_err_m=depth_err,
                hdg_err_deg=hdg_err,
                cross_track_m=g_out["cross_track_m"],
                u_mps=state["u_m_s"],
                v_mps=state["v_m_s"],
                pitch_deg=math.degrees(pitch_rad),
                sector_ranges_m=state["sector_ranges_m"],
            )
            t_target = teacher.query(
                depth_err_m=depth_err,
                hdg_err_deg=hdg_err,
                cross_track_m=g_out["cross_track_m"],
                u_mps=state["u_m_s"],
                v_mps=state["v_m_s"],
                pitch_deg=math.degrees(pitch_rad),
                sector_ranges_m=state["sector_ranges_m"],
            )
            if args.distill:
                student.train_step(feats, t_target)
                s_yaw, s_spd, s_pitch = student.predict(feats)
            else:
                s_yaw, s_spd, s_pitch = 0.0, 0.0, 0.0

            adj_hdg_sp = (g_out["heading_sp_deg"] + s_yaw * 30.0) % 360.0
            adj_speed_sp = max(0.35, g_out["speed_sp_mps"] * (1.0 + s_spd * 0.35))

            # 4) Classical Inner-Loop Control (`VehicleFlightController`)
            cmd = ctl.update(
                depth_m=state["depth_m"],
                pitch_deg=math.degrees(pitch_rad) - s_pitch * 4.0,
                roll_deg=math.degrees(roll_rad),
                heading_deg=hdg_deg,
                depth_sp_m=g_out["depth_sp_m"],
                heading_sp_deg=adj_hdg_sp,
                forward=g_out["forward_effort"],
                u_mps=state["u_m_s"] if state["dvl_locked"] else None,
                speed_sp_mps=adj_speed_sp if state["dvl_locked"] else None,
                now=now,
            )
            last_stern = cmd["stern"]

            for fin_name, defl in cmd["fins"].items():
                link.set_servo_pwm(FIN_OUTPUTS[fin_name], deflection_to_pwm(defl))
            link.set_servo_pwm(STERN_OUTPUT, effort_to_pwm(cmd["stern"]))
            link.set_servo_pwm(BOW_OUTPUT, effort_to_pwm(cmd["bow"]))

            if (now - t0) > MAX_MISSION_S:
                break
            if args.link == "real":
                time.sleep(1.0 / LOOP_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        neutral_all(link)
        link.disarm()


if __name__ == "__main__":
    main()
