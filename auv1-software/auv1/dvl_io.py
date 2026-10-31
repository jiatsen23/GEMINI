"""Commercial DVL I/O driver (Water Linked DVL-A50 / NMEA $WRZ compatible).

Follows the AUV1 Layering Rule (Section 6):
- Socket / serial / protocol parsing lives ONLY here (and in mavlink_io.py).
- Control, guidance, and navigation math (navigation.py, guidance.py,
  vehicle_control.py) never import socket or serial; they consume pure
  DvlReading dicts/dataclasses.
- Feature-flagged (`enabled=False` stub mode by default until the DVL is
  physically fitted on AUV1).

Supported commercial protocols:
1. Water Linked DVL-A50 JSON-over-TCP (default IP 192.168.2.95, port 16171)
   or UDP broadcast via BlueOS DVL extension.
2. NMEA-0183 / Water Linked ASCII serial (`$WRZ,vx,vy,vz,valid,alt,fom,cov*cs`).

Coordinate frame convention (matches AUV1 & ArduSub body frame):
- vx (u): forward (+nose), m/s
- vy (v): starboard (+right), m/s
- vz (w): down (+keel), m/s
- altitude_m: vertical distance to pool floor / seabed, m (+down)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time


@dataclass(frozen=True)
class DvlReading:
    """Single velocity + bottom-range measurement from a 4-beam Janus DVL."""
    t: float
    vx_m_s: float          # body surge u (+forward), m/s
    vy_m_s: float          # body sway v (+starboard), m/s
    vz_m_s: float          # body heave w (+down), m/s
    altitude_m: float      # range to bottom, m (-1.0 if invalid)
    fom_m_s: float         # Figure of Merit (velocity std-dev estimate, m/s)
    valid: bool            # True when >=3 beams have acoustic bottom lock

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "vx_m_s": self.vx_m_s,
            "vy_m_s": self.vy_m_s,
            "vz_m_s": self.vz_m_s,
            "altitude_m": self.altitude_m,
            "fom_m_s": self.fom_m_s,
            "valid": self.valid,
        }


def parse_waterlinked_json(line: str, now: float | None = None) -> DvlReading | None:
    """Parse a Water Linked DVL-A50 JSON velocity report (`type == 'velocity'`).

    Example payload from DVL-A50 TCP port 16171:
    {
        "time": 142.1, "vx": 0.62, "vy": -0.01, "vz": 0.03,
        "fom": 0.004, "altitude": 1.48, "velocity_valid": true,
        "status": 0, "type": "velocity"
    }
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    if msg.get("type") != "velocity":
        return None
    t = time.monotonic() if now is None else now
    return DvlReading(
        t=t,
        vx_m_s=float(msg.get("vx", 0.0)),
        vy_m_s=float(msg.get("vy", 0.0)),
        vz_m_s=float(msg.get("vz", 0.0)),
        altitude_m=float(msg.get("altitude", -1.0)),
        fom_m_s=float(msg.get("fom", 9.99)),
        valid=bool(msg.get("velocity_valid", False)),
    )


def parse_wrz_nmea(sentence: str, now: float | None = None) -> DvlReading | None:
    """Parse a Water Linked `$WRZ` NMEA sentence over UART/Serial.

    Format: $WRZ,vx,vy,vz,valid(y/n),altitude,fom,covariance,time_of_validity*CS
    """
    s = sentence.strip()
    if not s.startswith("$WRZ,"):
        return None
    if "*" in s:
        s, _ = s.split("*", 1)
    parts = s.split(",")
    if len(parts) < 7:
        return None
    try:
        t = time.monotonic() if now is None else now
        return DvlReading(
            t=t,
            vx_m_s=float(parts[1]),
            vy_m_s=float(parts[2]),
            vz_m_s=float(parts[3]),
            valid=(parts[4].lower() == "y"),
            altitude_m=float(parts[5]),
            fom_m_s=float(parts[6]),
        )
    except ValueError:
        return None


class DvlIO:
    """Non-blocking TCP JSON client for a commercial DVL (e.g. Water Linked A50).

    If `enabled=False` (default while DVL is on the roadmap), `read()`
    immediately returns `None` without opening network sockets, so mission
    scripts run unmodified with stub/fallback dead reckoning.
    """

    def __init__(self, host: str = "192.168.2.95", port: int = 16171,
                 enabled: bool = False, timeout: float = 0.2):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = ""
        self._latest: DvlReading | None = None

    def connect(self) -> bool:
        if not self.enabled:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            sock.setblocking(False)
            self._sock = sock
            return True
        except OSError:
            self._sock = None
            return False

    def poll(self, now: float | None = None) -> DvlReading | None:
        """Drain socket buffer and return the newest valid/invalid DVL frame."""
        if not self.enabled:
            return None
        if self._sock is None and not self.connect():
            return self._latest

        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    # Remote closed connection
                    self.close()
                    break
                self._buf += chunk.decode("utf-8", errors="ignore")
        except BlockingIOError:
            pass
        except OSError:
            self.close()

        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            reading = (parse_waterlinked_json(line, now=now)
                       or parse_wrz_nmea(line, now=now))
            if reading is not None:
                self._latest = reading
        return self._latest

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
