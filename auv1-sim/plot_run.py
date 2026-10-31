"""Plot a simulator run: depth, heading, pitch (with setpoints), fins, and
the top-down track (true vs. estimated).

    py plot_run.py                    # newest runs/*.json
    py plot_run.py runs/square_sim.json
    py plot_run.py --show             # also open a window

Saves <run>.png next to the JSON. Needs matplotlib.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        path = Path(args[0])
    else:
        runs = sorted((HERE / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not runs:
            sys.exit("no runs/*.json yet - run run_mission.py first")
        path = runs[-1]
    run = json.load(open(path))
    H = run["history"]
    pool = run["pool"]

    t = [r["truth"]["t"] for r in H]
    depth_true = [r["truth"]["pos"][2] for r in H]
    depth_est = [r["estimate"]["depth"] for r in H]
    depth_sp = [r.get("depth_sp", math.nan) for r in H]
    hdg_true = [math.degrees(r["truth"]["att"][2]) for r in H]
    hdg_sp = [r.get("heading_sp", math.nan) for r in H]
    pitch_true = [math.degrees(r["truth"]["att"][1]) for r in H]
    pitch_sp = [r.get("pitch_sp", math.nan) for r in H]
    fins = {n: [r["truth"]["fins_deg"][n] for r in H] for n in ("top", "bottom", "left", "right")}
    stern = [r["truth"]["stern_N"] for r in H]
    bow = [r["truth"]["bow_N"] for r in H]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.4, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, depth_true, label="true")
    ax.plot(t, depth_est, label="estimated", alpha=0.7)
    ax.plot(t, depth_sp, "k--", label="setpoint")
    ax.invert_yaxis(); ax.set_ylabel("depth (m)"); ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, hdg_true, label="true")
    ax.plot(t, hdg_sp, "k--", label="setpoint")
    ax.set_ylabel("heading (deg)"); ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, pitch_true, label="true")
    ax.plot(t, pitch_sp, "k--", label="cascade target")
    ax.set_ylabel("pitch (deg)"); ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[3, 0])
    for n, v in fins.items():
        ax.plot(t, v, label=n, lw=0.8)
    ax.set_ylabel("fin (deg)"); ax.set_xlabel("time (s)"); ax.legend(loc="upper right", fontsize=8, ncol=4)

    ax = fig.add_subplot(gs[0:3, 1])
    L, W = pool["length_m"], pool["width_m"]
    ax.add_patch(plt.Rectangle((-W / 2, -L / 2), W, L, fill=False, lw=2))
    ax.plot([r["truth"]["pos"][1] for r in H], [r["truth"]["pos"][0] for r in H], label="true track")
    ax.plot([r["estimate"]["pos_ne"][1] for r in H], [r["estimate"]["pos_ne"][0] for r in H],
            alpha=0.7, label="dead-reckoned")
    ax.set_aspect("equal"); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.legend(loc="upper right", fontsize=8); ax.set_title("top view")

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, stern, label="stern (N)"); ax.plot(t, bow, label="bow (N)")
    ax.set_xlabel("time (s)"); ax.legend(fontsize=8)

    fig.suptitle(f"{path.name} - mission '{run['mission']}'")
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
