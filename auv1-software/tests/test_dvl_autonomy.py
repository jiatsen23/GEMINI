"""Unit tests for DVL+Sonar SLAM, Natural Language Planning, and AI Teacher-Student Distillation.

Follows the AUV1 Layering Rule: all tests run deterministically in pure Python
with injected time and zero hardware / MAVLink dependencies.
"""

import math
from auv1.dvl_io import parse_waterlinked_json, parse_wrz_nmea
from auv1.guidance import (
    Waypoint,
    WaypointMissionRunner,
    los_heading_deg,
    resolve_depth_setpoint,
)
from auv1.llm_distill import (
    DistilledStudentPolicy,
    LLMTeacherSupervisor,
    build_feature_vector,
)
from auv1.navigation import DeadReckoningNavigator, lever_arm_correct
from auv1.nl_planner import NaturalLanguagePlanner
from auv1.sensor_fusion import DvlSonarFusionSLAM
from auv1.sonar_io import extract_range_from_ping360_profile
from auv1.vehicle_control import VehicleFlightController, bow_authority


def test_parse_waterlinked_json_and_wrz():
    raw_json = (
        '{"time": 12.5, "vx": 0.82, "vy": -0.04, "vz": 0.01, '
        '"fom": 0.004, "altitude": 1.45, "velocity_valid": true, '
        '"status": 0, "type": "velocity"}'
    )
    pkt = parse_waterlinked_json(raw_json, now=100.0)
    assert pkt is not None
    assert pkt.valid is True
    assert math.isclose(pkt.vx_m_s, 0.82)
    assert math.isclose(pkt.altitude_m, 1.45)

    nmea = "$WRZ,0.75,-0.02,0.01,y,1.30,0.005,0.001*5A"
    pkt_nmea = parse_wrz_nmea(nmea, now=101.0)
    assert pkt_nmea is not None
    assert pkt_nmea.valid is True
    assert math.isclose(pkt_nmea.vx_m_s, 0.75)
    assert math.isclose(pkt_nmea.altitude_m, 1.30)


def test_lever_arm_compensation_removes_pitch_and_yaw_induced_velocity():
    # DVL mounted at x=+0.20 m forward, z=+0.09 m down on keel.
    # Pure yaw rate r = +0.5 rad/s (nose swinging right) with zero CG speed
    # induces vy = r * rx = +0.10 m/s at the DVL transducer.
    r_dvl = (0.20, 0.0, 0.09)
    u, v, w = lever_arm_correct(0.0, 0.10, 0.0, 0.0, 0.0, 0.5, r_dvl)
    assert math.isclose(u, 0.0, abs_tol=1e-6)
    assert math.isclose(v, 0.0, abs_tol=1e-6)
    assert math.isclose(w, 0.0, abs_tol=1e-6)


def test_dead_reckoning_dvl_and_bumpless_fallback():
    nav = DeadReckoningNavigator(dvl_enabled=True, r_dvl_body_m=(0.2, 0.0, 0.09))
    nav.reset(north_m=0.0, east_m=0.0, depth_m=1.5)

    # Tick 0: initialize timestamp
    nav.update(
        now=0.0, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
        depth_m=1.5, stern_effort=0.5,
        dvl={"t": 0.0, "vx_m_s": 1.0, "vy_m_s": 0.0, "vz_m_s": 0.0,
             "altitude_m": 1.5, "fom_m_s": 0.005, "valid": True},
    )
    # Tick 1 (1.0 s later, heading North = 0 rad at 1.0 m/s): should advance +1.0 m North
    s1 = nav.update(
        now=1.0, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
        depth_m=1.5, stern_effort=0.5,
        dvl={"t": 1.0, "vx_m_s": 1.0, "vy_m_s": 0.0, "vz_m_s": 0.0,
             "altitude_m": 1.5, "fom_m_s": 0.005, "valid": True},
    )
    assert s1["dvl_locked"] is True
    assert math.isclose(s1["north_m"], 1.0, rel_tol=1e-3)

    # Tick 2: DVL loses bottom lock (valid=False) -> falls back to lagged thrust model
    s2 = nav.update(
        now=2.0, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
        depth_m=1.5, stern_effort=0.5,
        dvl={"t": 2.0, "vx_m_s": 0.0, "vy_m_s": 0.0, "vz_m_s": 0.0,
             "altitude_m": -1.0, "fom_m_s": 9.9, "valid": False},
    )
    assert s2["dvl_locked"] is False
    assert s2["north_m"] > 1.5  # continues advancing smoothly on fallback model


def test_los_guidance_and_floor_guard():
    # Path from (0,0) to (10,0) North; vehicle is 3 m East (+cross-track error)
    hdg_sp, xte, rem = los_heading_deg(5.0, 3.0, 0.0, 0.0, 10.0, 0.0, lookahead_m=3.0)
    assert math.isclose(xte, 3.0)
    assert math.isclose(rem, 5.0)
    # Should steer port (West of North, i.e., ~315 deg) to rejoin the North line
    assert math.isclose(hdg_sp, 315.0, abs_tol=1.0)

    # Floor guard: pool floor is at 1.5 m + 0.30 m = 1.80 m; min safe altitude = 0.35 m
    # Even if mission asks for 2.2 m depth, resolve_depth_setpoint clamps to 1.45 m
    clamped = resolve_depth_setpoint(
        target_depth_m=2.2, current_depth_m=1.5,
        altitude_m=0.30, min_safe_altitude_m=0.35,
    )
    assert math.isclose(clamped, 1.45)


def test_vehicle_flight_controller_speed_pid_and_water_speed_bow_blend():
    ctl = VehicleFlightController()
    out = ctl.update(
        depth_m=1.5, pitch_deg=0.0, roll_deg=0.0, heading_deg=0.0,
        depth_sp_m=1.5, heading_sp_deg=45.0,
        u_mps=0.1, speed_sp_mps=0.8, now=1.0,
    )
    assert out["stern"] > 0.3  # speed PID pushes stern thruster to accelerate
    # At low measured water speed (0.1 m/s), bow tunnel thruster retains high authority
    assert bow_authority(0.8, u_mps=0.1) > 0.8
    # At high measured water speed (0.9 m/s), bow tunnel thruster fades out completely
    assert bow_authority(0.1, u_mps=0.9) == 0.0


def test_ping360_profile_peak_extraction_and_sonar_slam_gate_detection():
    # Synthetic 100-bin Ping360 intensity profile with transmit ringdown in first 10 bins
    # and a strong acoustic echo at bin 40 (8.0 m at 20 m max range)
    bins = [250 if i < 6 else (220 if i == 40 else 15) for i in range(100)]
    rng_m, conf, valid = extract_range_from_ping360_profile(bins, max_range_m=20.0, min_range_m=0.75)
    assert valid is True
    assert math.isclose(rng_m, 8.1, abs_tol=0.25)
    assert conf > 0.8

    # Feed synthetic sonar hits on two gate posts at (8.0, -1.0) and (8.0, +1.0) into DvlSonarFusionSLAM
    slam = DvlSonarFusionSLAM(dvl_enabled=True, sonar_enabled=True)
    slam.reset(0.0, 0.0, 1.5)
    post_left_angle = math.degrees(math.atan2(-1.0, 8.0))
    post_right_angle = math.degrees(math.atan2(1.0, 8.0))
    post_range = math.hypot(8.0, 1.0)
    beams = [
        {"angle_deg": post_left_angle, "range_m": post_range, "confidence": 0.95, "valid": True},
        {"angle_deg": post_right_angle, "range_m": post_range, "confidence": 0.95, "valid": True},
    ]
    for step in range(4):
        state = slam.update(
            now=float(step) * 0.2,
            roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0,
            depth_m=1.5, stern_effort=0.4,
            dvl={"t": float(step) * 0.2, "vx_m_s": 0.0, "vy_m_s": 0.0, "vz_m_s": 0.0,
                 "altitude_m": 1.5, "fom_m_s": 0.004, "valid": True},
            sonar={"t": float(step) * 0.2, "max_range_m": 20.0, "beams": beams},
        )
    gates = [f for f in state["detected_features"] if f["kind"] == "gate"]
    assert len(gates) == 1
    assert math.isclose(gates[0]["north_m"], 8.0, abs_tol=0.6)
    assert math.isclose(gates[0]["east_m"], 0.0, abs_tol=0.6)


def test_natural_language_planner_compiles_missions():
    planner = NaturalLanguagePlanner(use_local_llm=False)
    res_map = planner.compile_command("I want to map out the pool at 1.6m depth")
    assert res_map["mission_type"] == "lawnmower_map"
    assert len(res_map["waypoints"]) >= 8
    assert math.isclose(res_map["waypoints"][0].depth_m, 1.6)

    # Gate command dynamically aligns with acoustically detected gate
    res_gate = planner.compile_command(
        "Find and pass through the gate",
        detected_features=[{"kind": "gate", "north_m": 8.0, "east_m": 0.5, "confidence": 0.92}],
    )
    assert res_gate["mission_type"] == "gate_passage"
    assert any(wp.label == "gate_center" and math.isclose(wp.east_m, 0.5) for wp in res_gate["waypoints"])


def test_teacher_student_dagger_distillation_convergence():
    teacher = LLMTeacherSupervisor(use_local_llm=False)
    student = DistilledStudentPolicy(seed=42)

    # Scenario: obstacle close ahead (sector 0 = 2.2 m), freer water to starboard
    slam_state = {
        "u_body_m_s": 0.8,
        "v_body_m_s": 0.0,
        "altitude_m": 1.4,
        "sector_clearance_m": [2.2, 6.0, 10.0, 12.0, 15.0, 12.0, 4.0, 3.0],
        "detected_features": [],
    }
    t_out = teacher.evaluate(slam_state, los_heading_deg=0.0, current_heading_deg=0.0, xte_m=0.0)
    assert t_out["avoidance_active"] is True
    assert abs(t_out["delta_heading_deg"]) > 10.0

    feats = build_feature_vector(
        u_mps=0.8, v_mps=0.0, xte_m=0.0, hdg_err_deg=0.0,
        altitude_m=1.4, sector_clearance_m=slam_state["sector_clearance_m"],
    )
    target = (t_out["delta_heading_deg"], t_out["speed_scale"], t_out["delta_pitch_deg"])
    loss_initial = student.train_step(feats, target, lr=0.05)
    loss_final = loss_initial
    for _ in range(40):
        loss_final = student.train_step(feats, target, lr=0.05)
    assert loss_final < loss_initial * 0.2
