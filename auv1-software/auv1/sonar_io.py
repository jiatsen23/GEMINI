"""Commercial Scanning / Multibeam Sonar I/O Driver (Blue Robotics Ping360 & Oculus).

Follows the AUV1 Layering Rule (Section 6):
- Hardware protocol parsing (UDP / Serial `brping` packets or JSON sonar fans)
  lives ONLY here.
- Feature-flagged (`enabled=False` by default while sonar is a roadmap sensor).
- Outputs pure `SonarRay` / `SonarScan` structures consumed by `sensor_fusion.py`.

How a Commercial Scanning Sonar (Blue Robotics Ping360) Works:
- Mechanically steps a single acoustic transducer through 400 gradians (360 deg,
  1 gradian = 0.9 deg) or a forward sector (e.g. +/-60 deg).
- At each angle, emits a 750 kHz acoustic pulse and records an array of 200-1200
  backscatter intensity samples (8-bit, 0..255) over the configured range
  (e.g. 2 m to 30 m).
- Strong echoes after the transmit ring-down indicate walls, gate pipes, or
  buoys. By thresholding and peak-finding the intensity profile (or using the
  full profile for log-odds inverse sensor modelling), we extract both range
  and acoustic cross-section intensity.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import time


@dataclass(frozen=True)
class SonarBeam:
    """Single acoustic beam measurement in the vehicle body frame."""
    angle_body_rad: float     # 0 = nose (+x), +pi/2 = starboard (+y)
    range_m: float            # first strong return range (m), or max_range_m if clear
    max_range_m: float        # configured sonar range window (m)
    intensity: float          # normalized return strength 0.0..1.0 (0 = no hit)
    hit: bool                 # True if an obstacle/target echo exceeded threshold


@dataclass(frozen=True)
class SonarScan:
    """A single beam or multi-beam sector scan at timestamp `t`."""
    t: float
    beams: tuple[SonarBeam, ...]

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "beams": [
                {
                    "angle_body_rad": b.angle_body_rad,
                    "range_m": b.range_m,
                    "max_range_m": b.max_range_m,
                    "intensity": b.intensity,
                    "hit": b.hit,
                }
                for b in self.beams
            ],
        }


def extract_range_from_ping360_profile(
    profile_bytes: bytes | list[int],
    max_range_m: float,
    blanking_m: float = 0.35,
    threshold: int = 110,
) -> tuple[float, float, bool]:
    """Convert a Ping360 8-bit acoustic intensity array into (range_m, intensity, hit).

    Ignores the transducer ring-down inside `blanking_m` and finds the strongest
    first return peak above `threshold` (0..255).
    """
    n = len(profile_bytes)
    if n == 0 or max_range_m <= 0.0:
        return max_range_m, 0.0, False

    min_idx = max(1, int((blanking_m / max_range_m) * n))
    best_idx = -1
    best_val = 0

    for i in range(min_idx, n):
        val = int(profile_bytes[i])
        if val >= threshold:
            # Walk to the local peak of this echo cluster
            best_idx = i
            best_val = val
            for j in range(i + 1, min(n, i + 12)):
                if int(profile_bytes[j]) >= best_val:
                    best_val = int(profile_bytes[j])
                    best_idx = j
                else:
                    break
            break

    if best_idx < 0:
        return max_range_m, 0.0, False

    range_m = ((best_idx + 0.5) / n) * max_range_m
    return range_m, min(1.0, best_val / 255.0), True


class SonarIO:
    """Non-blocking UDP driver for a BlueOS Ping360 / multibeam sonar stream.

    Defaults to `enabled=False` (stub mode) until the sonar hardware is fitted.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 12345,
                 enabled: bool = False):
        self.host = host
        self.port = port
        self.enabled = enabled
        self._sock: socket.socket | None = None
        self._latest: SonarScan | None = None

    def bind(self) -> bool:
        if not self.enabled:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.host, self.port))
            sock.setblocking(False)
            self._sock = sock
            return True
        except OSError:
            self._sock = None
            return False

    def poll(self, now: float | None = None) -> SonarScan | None:
        if not self.enabled:
            return None
        if self._sock is None and not self.bind():
            return self._latest

        t = time.monotonic() if now is None else now
        try:
            while True:
                data, _ = self._sock.recvfrom(8192)
                msg = json.loads(data.decode("utf-8", errors="ignore"))
                beams = []
                if "profile" in msg:
                    # Raw Ping360 gradian step (0..399, 200 = straight ahead)
                    grad = float(msg.get("angle_grad", 200.0))
                    ang_rad = math.radians((grad - 200.0) * 0.9)
                    max_r = float(msg.get("max_range_m", 15.0))
                    r_m, inten, hit = extract_range_from_ping360_profile(
                        msg["profile"], max_range_m=max_r
                    )
                    beams.append(SonarBeam(ang_rad, r_m, max_r, inten, hit))
                elif "beams" in msg:
                    for b in msg["beams"]:
                        beams.append(
                            SonarBeam(
                                angle_body_rad=float(b["angle_body_rad"]),
                                range_m=float(b["range_m"]),
                                max_range_m=float(b.get("max_range_m", 15.0)),
                                intensity=float(b.get("intensity", 0.8)),
                                hit=bool(b.get("hit", True)),
                            )
                        )
                if beams:
                    self._latest = SonarScan(t=t, beams=tuple(beams))
        except (BlockingIOError, json.JSONDecodeError, OSError):
            pass
        return self._latest

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
