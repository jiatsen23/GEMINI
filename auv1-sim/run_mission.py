"""Fly a mission with the REAL AUV1 control code against the simulator.

    py run_mission.py                       # sim, default mission, writes runs/
    py run_mission.py --mission square      # pick a mission by name
    py run_mission.py --pool config/pool.json --vehicle config/vehicle.json
    py run_mission.py --link real           # same code, real Pixhawk via BlueOS

What runs here is `auv1.vehicle_control.VehicleFlightController` - the
pitch-and-drive cascade written for the real torpedo - plus
`auv1.fin_mixer`, imported UNCHANGED from the auv1-software repo next door.
The only thing the sim replaces is the vehicle itself (SimLink instead of
MavlinkIO). That is the swap the "--link real" flag makes.

Outputs (in runs/):
    <name>.csv    one row per control tick - opens in Excel / plot_run.py
    <name>.json   everything, for viewer/viewer.html
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ── find the auv1 package (the real control code) ─────────────────────
def import_auv1(auv1_path: str | None):
    """Make `import auv1` work. Looks in: --auv1-path, then ../auv1-software,
    then whatever is pip-installed."""
    candidates = [Path(auv1_path)] if auv1_path else []
    candidates.append(HERE.parent / "auv1-software")
    for c in candidates:
        if (c / "auv1" / "__init__.py").exists():
            sys.path.insert(0, str(c))
            break
    try:
        import auv1  # noqa: F401
    except ImportError:
        sys.exit("Can't find the auv1 package. Put auv1-sim next to "
                 "auv1-software, or pass --auv1-path C:\\path\\to\\auv1-software")


# ── missions ──────────────────────────────────────────────────────────
# Each step: (name, targets, done_when). Targets not given keep the previous
# value. Depth control is pitch-and-drive, so every step that needs depth
# also needs forward speed (no VBS yet - see vehicle_control.py HONESTY BOX).
CRUISE = 0.6      # stern effort while diving/climbing (the fins need speed to bite)
LEG = 0.6         # stern effort on straight legs (pitch-and-drive needs speed: slow = lose depth)
TURN = 0.15       # stern effort while turning: slow enough that the bow tunnel
                  # thruster has authority (see bow_authority() in vehicle_control)
MISSIONS = {
    "square": [
        ("dive",    dict(depth=1.5, hdg=0.0, forward=CRUISE),   dict(depth_within=0.2, hold_s=2.0)),
        ("leg N",   dict(hdg=0.0, forward=LEG),                 dict(time_s=6.0)),
        ("turn E",  dict(hdg=90.0, forward=TURN),               dict(hdg_within=8.0, hold_s=1.0)),
        ("leg E",   dict(hdg=90.0, forward=LEG),                dict(time_s=6.0)),
        ("turn S",  dict(hdg=180.0, forward=TURN),              dict(hdg_within=8.0, hold_s=1.0)),
        ("leg S",   dict(hdg=180.0, forward=LEG),               dict(time_s=6.0)),
        ("turn W",  dict(hdg=270.0, forward=TURN),              dict(hdg_within=8.0, hold_s=1.0)),
        ("leg W",   dict(hdg=270.0, forward=LEG),               dict(time_s=6.0)),
        ("turn N",  dict(hdg=0.0, forward=TURN),                dict(hdg_within=8.0, hold_s=1.0)),
        ("surface", dict(depth=0.2, forward=CRUISE),            dict(depth_within=0.15, hold_s=2.0)),
        ("stop",    dict(forward=0.0),                          dict(time_s=4.0)),
    ],
    "depth_step": [
        ("dive 1 m",   dict(depth=1.0, hdg=0.0, forward=CRUISE), dict(time_s=12.0)),
        ("dive 2 m",   dict(depth=2.0),                          dict(time_s=12.0)),
        ("turn back",  dict(hdg=180.0, forward=TURN),            dict(hdg_within=8.0, hold_s=1.0)),
        ("up to 1 m",  dict(depth=1.0, forward=CRUISE),          dict(time_s=12.0)),
        ("surface",    dict(depth=0.2),                          dict(depth_within=0.15, hold_s=2.0)),
        ("stop",       dict(forward=0.0),                        dict(time_s=3.0)),
    ],
    "heading_only": [
        ("north",      dict(depth=0.0, hdg=0.0, forward=LEG),   dict(time_s=8.0)),
        ("fins east",  dict(hdg=90.0),                          dict(time_s=15.0)),
        ("bow south",  dict(hdg=180.0, forward=TURN),           dict(time_s=15.0)),
        ("stop",       dict(forward=0.0),                       dict(time_s=3.0)),
    ],
}

LOOP_HZ = 10
MAX_MISSION_S = 300.0

# Gains that behave in the SIM (enable with --tuned). The defaults in
# auv1/vehicle_control.py are the repo's untouched starting guesses; these
# are what the simulator suggested after watching them: the fins chattered
# (derivative terms amplifying sensor noise through the cascade) and pitch /
# heading settled a few degrees short of target (no integral term).
# Treat as a hint for the pool, not as truth.
TUNED_GAINS = {
    "depth_to_pitch": dict(kp=1.0, ki=0.05, kd=0.1, output_limit=1.0, integral_limit=0.3),
    "pitch":          dict(kp=0.04, ki=0.03, kd=0.004, output_limit=1.0, integral_limit=20.0),
    "roll":           dict(kp=0.02, ki=0.0, kd=0.002, output_limit=0.5, integral_limit=10.0),
    "heading":        dict(kp=0.02, ki=0.002, kd=0.005, output_limit=0.6, integral_limit=60.0),
}


def effort_to_pwm(effort: float) -> int:
    return int(1500 + max(-1.0, min(1.0, effort)) * 400)


def step_done(done, t_in_step, depth_err, hdg_err, stable_since, now):
    """Same idea as scripts/mission_square.py: timed steps just wait, tolerance
    steps must stay inside tolerance for hold_s seconds."""
    if "time_s" in done:
        return t_in_step >= done["time_s"], None
    ok = True
    if "depth_within" in done:
        ok = ok and abs(depth_err) <= done["depth_within"]
    if "hdg_within" in done:
        ok = ok and abs(hdg_err) <= done["hdg_within"]
    if not ok:
        return False, None
    if stable_since is None:
        return False, now
    return (now - stable_since) >= done.get("hold_s", 0.0), stable_since


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="square", choices=MISSIONS.keys())
    ap.add_argument("--link", default="sim", choices=["sim", "real"])
    ap.add_argument("--connection", default="udpin:0.0.0.0:14551",
                    help="MAVLink endpoint when --link real")
    ap.add_argument("--pool", default=None, help="pool JSON (default config/pool.json)")
    ap.add_argument("--vehicle", default=None, help="vehicle JSON (default config/vehicle.json)")
    ap.add_argument("--auv1-path", default=None)
    ap.add_argument("--seed", type=int, default=1, help="sensor-noise seed")
    ap.add_argument("--name", default=None, help="output file stem (default: mission name)")
    ap.add_argument("--tuned", action="store_true", help="use TUNED_GAINS instead of the repo defaults")
    args = ap.parse_args()

    import_auv1(args.auv1_path)
    from auv1.vehicle_control import VehicleFlightController
    from auv1.fin_mixer import deflection_to_pwm
    from auv1.controllers import wrap_error_deg

    # ── the ONE line that differs between simulation and the real vehicle ──
    if args.link == "sim":
        from auv1sim.sim_link import SimLink
        link = SimLink(vehicle_json=args.vehicle, pool_json=args.pool, seed=args.seed)
        clock = link.now                    # simulated seconds
        outputs = link.vehicle_cfg          # servo numbering from vehicle.json
        fin_out = outputs["fins"]["outputs"]
        stern_out = outputs["stern_thruster"]["servo_output"]
        bow_out = outputs["bow_thruster"]["servo_output"]
    else:
        from auv1.mavlink_io import MavlinkIO
        link = MavlinkIO(args.connection)
        clock = time.monotonic
        fin_out = {"top": 4, "bottom": 5, "left": 6, "right": 7}
        stern_out, bow_out = 8, 9
        print("waiting for heartbeat...")
        if not link.wait_heartbeat():
            sys.exit("no heartbeat")

    ctl = VehicleFlightController()
    if args.tuned:
        from auv1.controllers import PID
        for loop, gains in TUNED_GAINS.items():
            setattr(ctl, loop, PID(**gains))
    link.arm()

    mission = MISSIONS[args.mission]
    targets = dict(depth=0.0, hdg=0.0, forward=0.0)
    rows = []
    t0 = clock()
    step_i, step_t0, stable_since = 0, t0, None
    print(f"mission '{args.mission}' on {args.link}: {len(mission)} steps")

    while step_i < len(mission):
        name, new_targets, done = mission[step_i]
        targets.update(new_targets)

        att = link.get_attitude()
        if att is None:
            print("no attitude - skipping tick")
            continue
        roll, pitch, yaw = (math.degrees(a) for a in att)
        depth = link.get_depth()
        now = clock()

        cmd = ctl.update(depth_m=depth, pitch_deg=pitch, roll_deg=roll,
                         heading_deg=yaw % 360.0, depth_sp_m=targets["depth"],
                         heading_sp_deg=targets["hdg"], forward=targets["forward"],
                         now=now)

        for fin, defl in cmd["fins"].items():
            link.set_servo_pwm(fin_out[fin], deflection_to_pwm(defl))
        link.set_servo_pwm(stern_out, effort_to_pwm(cmd["stern"]))
        link.set_servo_pwm(bow_out, effort_to_pwm(cmd["bow"]))

        depth_err = targets["depth"] - depth
        hdg_err = wrap_error_deg(targets["hdg"], yaw % 360.0)
        row = dict(t=round(now - t0, 3), step=name, depth=depth, depth_sp=targets["depth"],
                   heading=yaw % 360.0, heading_sp=targets["hdg"], pitch=pitch,
                   pitch_sp=cmd["pitch_sp"], roll=roll, stern=cmd["stern"], bow=cmd["bow"],
                   **{f"fin_{k}": v for k, v in cmd["fins"].items()})
        rows.append(row)
        if args.link == "sim":
            link.annotate(step=name, depth_sp=targets["depth"], heading_sp=targets["hdg"],
                          pitch_sp=cmd["pitch_sp"], forward=targets["forward"])

        finished, stable_since = step_done(done, now - step_t0, depth_err, hdg_err,
                                           stable_since, now)
        if finished:
            print(f"  [{now - t0:6.1f}s] step '{name}' done  depth={depth:.2f} hdg={yaw % 360:.0f}")
            step_i, step_t0, stable_since = step_i + 1, now, None
        if now - t0 > MAX_MISSION_S:
            print("mission timed out")
            break
        if args.link == "real":
            time.sleep(1.0 / LOOP_HZ)

    link.disarm()

    # ── save ──────────────────────────────────────────────────────────
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    stem = args.name or f"{args.mission}_{args.link}"
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    if args.link == "sim":
        json_path = out_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump({"mission": args.mission, "pool": link.pool_cfg,
                       "vehicle": link.vehicle_cfg, "history": link.history}, f)
        print(f"wrote {json_path}  ->  open viewer/viewer.html and load it")


if __name__ == "__main__":
    main()
