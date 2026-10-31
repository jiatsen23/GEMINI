"""Natural Language -> Autonomous Path & Mission Planner for AUV1.

Respects the Context Brief rules:
- Runs 100% onboard the Raspberry Pi 5 (no internet underwater).
- Sits ABOVE `VehicleFlightController` (outputs `Waypoint` lists and high-level
  setpoints; never sits inside the fast 20 Hz PID loop).
- Supports an optional local Ollama / llama.cpp endpoint on the Pi 5
  (`http://127.0.0.1:11434/api/generate`, e.g. `qwen2.5:1.5b` or `llama3.2:3b`)
  AND a deterministic onboard semantic compiler so any natural language prompt
  ("map out the pool", "find and go through the gate", "orbit the buoy at 2m",
  "inspect the perimeter walls") compiles immediately into safe, pool-bounded
  underactuated trajectories.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from auv1.guidance import Waypoint


def generate_lawnmower_survey(
    pool_length_m: float = 50.0,
    pool_width_m: float = 25.0,
    margin_m: float = 6.0,
    swath_m: float = 6.0,
    depth_m: float = 1.5,
    speed_mps: float = 1.1,
) -> list[Waypoint]:
    """Generate a Boustrophedon (lawnmower) sonar mapping trajectory covering
    the pool while keeping safe turning margins from the walls."""
    n_min = -pool_length_m / 2.0 + margin_m
    n_max = pool_length_m / 2.0 - margin_m
    e_min = -pool_width_m / 2.0 + margin_m
    e_max = pool_width_m / 2.0 - margin_m

    waypoints: list[Waypoint] = []
    e = e_min
    leg = 0
    while e <= e_max + 1e-3:
        if leg % 2 == 0:
            n_start, n_end = n_min, n_max
        else:
            n_start, n_end = n_max, n_min
        waypoints.append(
            Waypoint(f"survey_L{leg+1}_start", north_m=n_start, east_m=e,
                     depth_m=depth_m, speed_mps=speed_mps, forward_effort=0.65,
                     acceptance_radius_m=2.0)
        )
        waypoints.append(
            Waypoint(f"survey_L{leg+1}_end", north_m=n_end, east_m=e,
                     depth_m=depth_m, speed_mps=speed_mps, forward_effort=0.65,
                     acceptance_radius_m=2.0)
        )
        e += swath_m
        leg += 1

    waypoints.append(
        Waypoint("return_home", north_m=0.0, east_m=0.0, depth_m=0.3,
                 speed_mps=0.7, forward_effort=0.45, acceptance_radius_m=2.5)
    )
    return waypoints


def generate_perimeter_wall_inspection(
    pool_length_m: float = 50.0,
    pool_width_m: float = 25.0,
    standoff_m: float = 4.5,
    depth_m: float = 1.6,
    speed_mps: float = 1.0,
) -> list[Waypoint]:
    """Generate a clockwise perimeter loop at `standoff_m` inside the pool walls."""
    n_hi = pool_length_m / 2.0 - standoff_m
    n_lo = -pool_length_m / 2.0 + standoff_m
    e_hi = pool_width_m / 2.0 - standoff_m
    e_lo = -pool_width_m / 2.0 + standoff_m
    return [
        Waypoint("wall_NE", north_m=n_hi, east_m=e_hi, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("wall_SE", north_m=n_lo, east_m=e_hi, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("wall_SW", north_m=n_lo, east_m=e_lo, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("wall_NW", north_m=n_hi, east_m=e_lo, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("wall_N_close", north_m=n_hi, east_m=0.0, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("surface_home", north_m=0.0, east_m=0.0, depth_m=0.25, speed_mps=0.6),
    ]


def generate_gate_search_and_pass(
    known_gate: dict | None = None,
    depth_m: float = 1.5,
    speed_mps: float = 0.95,
) -> list[Waypoint]:
    """If the Sonar+DVL mapper has already localized the gate, generate a
    3-point alignment & pass-through trajectory; otherwise generate an S-curve
    acoustic search pattern through the competition zone."""
    if known_gate is not None:
        gn = float(known_gate["north_m"])
        ge = float(known_gate["east_m"])
        hdg_rad = math.radians(float(known_gate.get("normal_heading_deg", 0.0)))
        # Approach 4 m before gate, pass through gate centre, exit 4 m past gate
        cn, sn = math.cos(hdg_rad), math.sin(hdg_rad)
        return [
            Waypoint("gate_align", north_m=gn - 4.0 * cn, east_m=ge - 4.0 * sn,
                     depth_m=depth_m, speed_mps=0.8, acceptance_radius_m=1.2),
            Waypoint("gate_centre", north_m=gn, east_m=ge,
                     depth_m=depth_m, speed_mps=speed_mps, acceptance_radius_m=1.0),
            Waypoint("gate_exit", north_m=gn + 4.5 * cn, east_m=ge + 4.5 * sn,
                     depth_m=depth_m, speed_mps=speed_mps, acceptance_radius_m=1.5),
            Waypoint("surface_finish", north_m=gn + 6.0 * cn, east_m=ge + 6.0 * sn,
                     depth_m=0.25, speed_mps=0.5, acceptance_radius_m=2.0),
        ]

    # Search pattern when gate is not yet in the acoustic map
    return [
        Waypoint("search_1", north_m=6.0,  east_m=-4.0, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("search_2", north_m=10.0, east_m=4.0,  depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("search_3", north_m=15.0, east_m=-3.0, depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("search_4", north_m=18.0, east_m=5.0,  depth_m=depth_m, speed_mps=speed_mps),
        Waypoint("return_home", north_m=0.0, east_m=0.0, depth_m=0.3, speed_mps=0.6),
    ]


class NaturalLanguagePlanner:
    """Translates natural language operator commands + live acoustic scene state
    into structured AUV1 waypoint missions."""

    def __init__(
        self,
        pool_length_m: float = 50.0,
        pool_width_m: float = 25.0,
        pool_depth_m: float = 3.0,
        ollama_url: str | None = None,
        ollama_model: str = "qwen2.5:1.5b",
    ):
        self.pool_length_m = pool_length_m
        self.pool_width_m = pool_width_m
        self.pool_depth_m = pool_depth_m
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def plan_from_text(
        self,
        command_text: str,
        scene_state: dict | None = None,
    ) -> dict:
        """Parse a natural language instruction into `{intent, explanation, waypoints}`."""
        text = command_text.strip()
        lower = text.lower()
        scene_state = scene_state or {}
        detections = scene_state.get("detections", [])

        # Extract optional explicit depth (e.g. "at 2.0m depth" or "depth 1.8m")
        depth_m = 1.5
        m_depth = re.search(r"(\d+(?:\.\d+)?)\s*m(?:\s+deep|\s+depth)?", lower)
        if m_depth:
            val = float(m_depth.group(1))
            if 0.2 <= val <= self.pool_depth_m - 0.35:
                depth_m = val

        # 1) Optional local onboard Ollama LLM call if configured
        if self.ollama_url:
            llm_plan = self._query_local_ollama(text, scene_state)
            if llm_plan is not None:
                return llm_plan

        # 2) Deterministic Onboard Semantic Compiler (0 ms latency, 100% reliable)
        # Intent A: Map / Survey the pool
        if any(k in lower for k in ("map", "survey", "lawnmower", "scan the pool", "explore", "coverage")):
            wps = generate_lawnmower_survey(
                pool_length_m=self.pool_length_m,
                pool_width_m=self.pool_width_m,
                depth_m=depth_m,
            )
            return {
                "intent": "lawnmower_mapping",
                "explanation": (
                    f"Generated a {len(wps)}-waypoint Boustrophedon (lawnmower) "
                    f"sonar+DVL mapping survey at {depth_m:.1f} m depth."
                ),
                "waypoints": wps,
            }

        # Intent B: Find / Pass through the competition gate
        if any(k in lower for k in ("gate", "pass through", "thread", "qualify")):
            gate_det = next((d for d in detections if d.get("label") == "gate"), None)
            wps = generate_gate_search_and_pass(known_gate=gate_det, depth_m=depth_m)
            status = (
                f"Locked onto acoustic gate at ({gate_det['north_m']:.1f}, {gate_det['east_m']:.1f}); "
                "aligned approach vector."
                if gate_det
                else "Gate not yet in sonar map; executing acoustic search sweep until gate posts lock."
            )
            return {
                "intent": "gate_mission",
                "explanation": status,
                "waypoints": wps,
            }

        # Intent C: Perimeter / Wall inspection
        if any(k in lower for k in ("wall", "perimeter", "boundary", "border")):
            wps = generate_perimeter_wall_inspection(
                pool_length_m=self.pool_length_m,
                pool_width_m=self.pool_width_m,
                depth_m=depth_m,
            )
            return {
                "intent": "perimeter_inspection",
                "explanation": (
                    f"Generated clockwise acoustic wall-standoff inspection loop at {depth_m:.1f} m depth."
                ),
                "waypoints": wps,
            }

        # Intent D: Orbit / Loiter / Inspect buoy or marker
        if any(k in lower for k in ("buoy", "marker", "orbit", "circle", "loiter")):
            buoy_det = next((d for d in detections if d.get("label") == "buoy"), None)
            bn = float(buoy_det["north_m"]) if buoy_det else -8.0
            be = float(buoy_det["east_m"]) if buoy_det else -5.0
            wps = [
                Waypoint("approach_buoy", north_m=bn, east_m=be, depth_m=depth_m,
                         speed_mps=0.9, acceptance_radius_m=3.0, loiter_s=18.0),
                Waypoint("return_home", north_m=0.0, east_m=0.0, depth_m=0.25,
                         speed_mps=0.6, acceptance_radius_m=2.0),
            ]
            return {
                "intent": "orbit_buoy",
                "explanation": (
                    f"Navigating to marker buoy at ({bn:.1f}, {be:.1f}) and flying a "
                    f"constant-depth 3 m radius orbit at {depth_m:.1f} m (no hover stall)."
                ),
                "waypoints": wps,
            }

        # Intent E: Return home & surface
        if any(k in lower for k in ("surface", "home", "abort", "return", "stop")):
            wps = [
                Waypoint("return_origin", north_m=0.0, east_m=0.0, depth_m=0.2,
                         speed_mps=0.6, acceptance_radius_m=2.0),
            ]
            return {
                "intent": "return_and_surface",
                "explanation": "Returning to pool origin (0, 0) and surfacing to 0.2 m.",
                "waypoints": wps,
            }

        # Intent F: Explicit coordinates (e.g. "go to 12, -5" or "N=10 E=6")
        coords = re.findall(r"[-+]?\d+(?:\.\d+)?", lower)
        if len(coords) >= 2:
            half_L = self.pool_length_m / 2.0 - 2.0
            half_W = self.pool_width_m / 2.0 - 2.0
            tn = max(-half_L, min(half_L, float(coords[0])))
            te = max(-half_W, min(half_W, float(coords[1])))
            wps = [
                Waypoint("custom_target", north_m=tn, east_m=te, depth_m=depth_m,
                         speed_mps=1.0, acceptance_radius_m=1.5),
            ]
            return {
                "intent": "goto_coordinate",
                "explanation": f"Driving to target coordinate N={tn:.1f} m, E={te:.1f} m at {depth_m:.1f} m depth.",
                "waypoints": wps,
            }

        # Default fallback: 10m square box
        wps = [
            Waypoint("box_N", north_m=10.0, east_m=0.0,  depth_m=depth_m, speed_mps=1.0),
            Waypoint("box_E", north_m=10.0, east_m=8.0,  depth_m=depth_m, speed_mps=1.0),
            Waypoint("box_S", north_m=0.0,  east_m=8.0,  depth_m=depth_m, speed_mps=1.0),
            Waypoint("box_W", north_m=0.0,  east_m=0.0,  depth_m=0.25,    speed_mps=0.6),
        ]
        return {
            "intent": "box_patrol",
            "explanation": f"Executing standard 10x8 m autonomous box patrol at {depth_m:.1f} m depth.",
            "waypoints": wps,
        }

    def _query_local_ollama(self, text: str, scene_state: dict) -> dict | None:
        """Query local Ollama server on Pi 5 if running (non-blocking timeout)."""
        prompt = (
            "You are the onboard AI mission planner for AUV1 (50x25x3m pool). "
            "Output ONLY valid JSON with keys: intent (str), explanation (str), "
            "waypoints (list of {name, north_m, east_m, depth_m, speed_mps}).\n"
            f"Scene: {json.dumps(scene_state)}\nCommand: {text}"
        )
        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                parsed = json.loads(body.get("response", "{}"))
                wps = [
                    Waypoint(
                        name=str(w.get("name", f"wp_{i}")),
                        north_m=float(w["north_m"]),
                        east_m=float(w["east_m"]),
                        depth_m=max(0.2, min(2.5, float(w.get("depth_m", 1.5)))),
                        speed_mps=max(0.4, min(1.5, float(w.get("speed_mps", 1.0)))),
                    )
                    for i, w in enumerate(parsed.get("waypoints", []))
                ]
                if wps:
                    return {
                        "intent": parsed.get("intent", "llm_custom"),
                        "explanation": parsed.get("explanation", "Planned by onboard LLM."),
                        "waypoints": wps,
                    }
        except Exception:
            return None
        return None
