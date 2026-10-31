# auv1-sim — AUV1 in a virtual pool

A small software-in-the-loop simulator for AUV1. It stands in for the wet
vehicle so the real control code in `auv1-software` can be run, watched and
tuned before the hull is finished.

```
                  ┌──────────────────────── SIMULATOR (this folder) ────────────────────────┐
                  │  actuators ──► physics (6-DOF torpedo in a pool) ──► noisy sensors        │
                  │  fins/T200s      dynamics.py                        depth, IMU, compass,  │
                  │                                                     GPS only at surface   │
                  └───────▲───────────────────────────────────────────────────┬──────────────┘
                          │ set_servo_pwm()                                   │ get_attitude()
                          │                                                   │ get_depth()
                          │              SimLink  (looks exactly like MavlinkIO)               │
                          │                                                   ▼
                  ┌───────┴─────────────────── REAL CONTROL CODE (auv1-software) ─────────────┐
                  │  estimator (in sim: estimator.py; on vehicle: ArduSub EKF in the Pixhawk)  │
                  │  VehicleFlightController  ── depth→pitch cascade, heading, roll            │
                  │  fin_mixer                ── "+" fin mixing, PWM                           │
                  └────────────────────────────────────────────────────────────────────────────┘
```

When the vehicle is ready, `run_mission.py --link real` replaces `SimLink`
with `MavlinkIO` and nothing else changes.

## Quick start (Windows)

```
cd C:\projects\auv1-sim
py -m pip install numpy matplotlib
py run_mission.py --tuned                 # square mission, writes runs\square_sim.*
py plot_run.py                            # runs\square_sim.png
py bake_viewer.py runs\square_sim.json    # runs\square_sim_viewer.html  -> open in a browser
```

`auv1-sim` must sit next to `auv1-software` (it imports the `auv1` package
from there), or pass `--auv1-path`.

Other missions: `--mission depth_step`, `--mission heading_only`. Sensor noise
is seeded (`--seed 2` for a different run). `--tuned` uses the gains in
`TUNED_GAINS` (top of `run_mission.py`); without it you get the untouched
starting-guess gains from `vehicle_control.py` — try both, the difference is
the point.

## The 3D viewer

`viewer/viewer.html` is a single self-contained file (three.js inlined, no
internet needed). Open it, load a `runs/*.json`, press play. Orange = true
track, blue wireframe = where the vehicle *thinks* it is (dead reckoning, no
GPS underwater). `bake_viewer.py` produces a copy with the run (and STL)
embedded so it can be sent to anyone as one file.

### Using the CAD

Export the general assembly from SOLIDWORKS as one STL: open
`AUV1P-HULL-A001-V1 General Assembly.SLDASM` → File → Save As → STL →
Options → tick **"Save all components of an assembly in a single file"**,
units mm, coarse resolution is fine. Load it with the "CAD (STL)" picker (or
pass it as the second argument to `bake_viewer.py`), then set "model nose
points" / "model up" to whichever SOLIDWORKS axes the nose and the top of the
hull lie along. Scale 0.001 converts mm to metres. Until then the viewer draws
a procedural torpedo from `config/vehicle.json` (that one animates its fins;
the STL is rigid).

## Customising

* `config/pool.json` — size, depth, start position and heading, wall stiffness.
* `config/vehicle.json` — mass, volume, metacentric height, drag, thruster
  limits, fin geometry, sensor noise/bias, telemetry rate. Every number is a
  first estimate; replace with measured values (float test, thruster bench,
  fin blade dimensions) as they arrive.
* Missions — the `MISSIONS` dict in `run_mission.py`.

## What the simulator found on day one

These came out of running the repo's current `vehicle_control.py` against the
model. None of the repo files were changed — they are for you to decide on.

1. **`fin_mixer.SIGNS` is internally inconsistent for left/right.** For any
   physical servo mounting, pitch and roll on the *right* fin must have the
   same sign and on the *left* fin opposite signs (nose-up and roll-right
   both push the right fin down; nose-up pushes the left fin down but
   roll-right pushes it up). The table currently has it the other way round,
   so the horizontal fins cancel the vertical fins' roll torque. Harmless
   until you want roll authority; the bench SIGNS check will hit it.
2. **Fin chatter.** With the default gains the fins flick ±20° at 10 Hz. The
   outer depth loop's D term turns depth-sensor noise into a noisy pitch
   target, and the inner pitch loop's D term amplifies that again. Servos
   will not thank you. `TUNED_GAINS` cuts the D terms; the proper fix is
   derivative-on-measurement or a low-pass on depth in `controllers.py`.
3. **Steady-state error** in pitch (~5°) and heading (~7°) with the defaults —
   no integral terms. Small `ki` on both loops fixed it in the sim.
4. **Pitch-and-drive needs speed.** Below ~1 m/s (stern effort < ~0.4) the
   fins cannot hold depth; holding a nose-down trim needs a stern-up fin
   force that *lifts* the vehicle, and at low speed that beats the dive
   component. Every slow bow-thruster turn in the square mission loses
   ~0.2 m of depth. Plan missions accordingly, or this is the argument for
   the VBS.
5. **Turning radius.** With rudders only (bow thruster fades out above
   forward 0.4) the turn radius at 1 m/s is ~7 m — too wide for a 25 m pool
   square. The bow tunnel thruster at low speed turns it in ~8 s. The mission
   therefore slows down to turn.

## Model limits (v1)

Diagonal added mass, no hull lift/Munk moment, no propeller torque, no
currents, no VBS, no sensor latency. Fin lift includes both deflection and
local flow angle (so the fins stabilise the vehicle like real ones do).
Physics at 200 Hz, telemetry/control at 10 Hz. Add what matters when a pool
run disagrees with the model — that comparison is the whole value of it.
