"""SimLink: a stand-in for auv1.mavlink_io.MavlinkIO.

This is the whole trick of software-in-the-loop. The control code talks to
an object with these methods:

    wait_heartbeat()  get_attitude()  get_depth()  set_servo_pwm()  arm()  disarm()
    get_dvl()         get_sonar()     get_gyro()   get_gps()

On the real vehicle that object is MavlinkIO (plus DvlIO and SonarIO when
fitted) and the methods go over MAVLink / Ethernet to the hardware. Here it is
SimLink and the methods go to the physics model.
"""

from . import config as cfg
from .dynamics import Vehicle
from .sensors import Sensors
from .estimator import Estimator

PHYSICS_DT = 0.005     # 200 Hz physics; the controller runs at telemetry_hz


class SimLink:
    def __init__(self, vehicle_json: str | None = None, pool_json: str | None = None,
                 seed: int = 1, dvl_enabled: bool | None = None,
                 sonar_enabled: bool | None = None):
        self.vehicle_cfg = cfg.load("vehicle", vehicle_json)
        if dvl_enabled is not None:
            self.vehicle_cfg["sensors"]["dvl_enabled"] = bool(dvl_enabled)
        if sonar_enabled is not None:
            self.vehicle_cfg["sensors"]["sonar_enabled"] = bool(sonar_enabled)
        self.pool_cfg = cfg.load("pool", pool_json)
        self.vehicle = Vehicle(self.vehicle_cfg, self.pool_cfg)
        self.sensors = Sensors(self.vehicle_cfg["sensors"], seed=seed)
        r_dvl = self.vehicle_cfg["sensors"].get("dvl_position_body_m", [0.2, 0.0, 0.09])
        self.estimator = Estimator(r_dvl_body_m=r_dvl)
        self.telemetry_dt = 1.0 / self.vehicle_cfg["sensors"]["telemetry_hz"]
        self.last_reading = self.sensors.read(self.vehicle)
        self.estimator.update(self.last_reading, 0.0)
        self.history = []        # one record per telemetry tick, for viewer/plots

    # ── the MavlinkIO look-alike API ───────────────────────────────────
    def wait_heartbeat(self, timeout: float = 10.0) -> bool:
        return True

    def get_attitude(self, timeout: float = 1.0):
        """Advance the world by one telemetry period; return ESTIMATED
        (roll, pitch, yaw) in radians - like ATTITUDE from the autopilot."""
        steps = max(1, round(self.telemetry_dt / PHYSICS_DT))
        for _ in range(steps):
            self.vehicle.step(PHYSICS_DT)
        self.last_reading = self.sensors.read(self.vehicle)
        self.estimator.update(self.last_reading, self.vehicle.cmd_stern)
        self._record()
        r, p, y = self.estimator.att
        return r, p, y

    def get_depth(self, timeout: float = 1.0):
        """ESTIMATED depth in metres, positive down (GLOBAL_POSITION_INT-like)."""
        return self.estimator.depth

    def get_dvl(self) -> dict | None:
        """Return latest simulated DVL packet (or None if dvl_enabled is False)."""
        return self.last_reading.get("dvl")

    def get_sonar(self) -> dict | None:
        """Return latest simulated Sonar scan (or None if sonar_enabled is False)."""
        return self.last_reading.get("sonar")

    def get_gyro(self) -> tuple[float, float, float]:
        """Return latest body angular rates (p, q, r) in rad/s."""
        g = self.last_reading.get("gyro_rad_s", [0.0, 0.0, 0.0])
        return float(g[0]), float(g[1]), float(g[2])

    def get_gps(self) -> list[float] | None:
        """Return surface GPS [north_m, east_m] or None when submerged."""
        return self.last_reading.get("gps_ne_m")

    def set_servo_pwm(self, output: int, pwm_us: int) -> None:
        self.vehicle.command_pwm(output, int(pwm_us))

    def set_mode(self, mode_name: str) -> bool:
        return True

    def arm(self, timeout: float = 5.0) -> bool:
        self.vehicle.armed = True
        return True

    def disarm(self) -> None:
        self.vehicle.armed = False

    def send_manual_control(self, *args, **kwargs):
        raise NotImplementedError(
            "AUV1 has no ArduSub mixer for its fins - drive it with "
            "set_servo_pwm() like the real vehicle, not MANUAL_CONTROL")

    # ── extras that only exist in the sim ──────────────────────────────
    def now(self) -> float:
        """Simulated time in seconds (use instead of time.monotonic())."""
        return self.vehicle.t

    def annotate(self, **fields):
        """Attach controller-side values (setpoints, step name...) to the
        latest history record so the viewer can show them."""
        if self.history:
            self.history[-1].update(fields)

    def _record(self):
        rec = {"truth": self.vehicle.truth(),
               "sensors": self.last_reading,
               "estimate": self.estimator.state()}
        self.history.append(rec)
