"""
controller.py — Turret orchestration: detect → align → aim → fire.

Trajectory lookup table maps measured distance to pitch angle and flywheel
throttle. Fill this in empirically: shoot at known distances, record what
settings land in the cup, and add entries to TRAJECTORY_TABLE.

Usage:
    python -m pi.controller            # auto loop
    python -m pi.controller --manual   # interactive shell
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Tuple

from .pico_comms import PicoComms
from .vision import CupDetector, DetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical / tuning constants  (adjust these as you calibrate)
# ---------------------------------------------------------------------------

CAMERA_HFOV_DEG = 60.0      # horizontal field of view of the camera
CAMERA_VFOV_DEG = 40.0      # vertical field of view

YAW_ALIGN_FREQ_HZ  = 300    # step freq during iterative yaw centering (slower = more precise)
YAW_COARSE_FREQ_HZ = 600    # step freq for large initial moves
PITCH_FREQ_HZ      = 400    # step freq for pitch moves

YAW_DEAD_ZONE   = 0.04      # |x_norm - AIM_X_OFFSET| below this = centered enough to fire
AIM_X_OFFSET    = 0.0       # x_norm when shooter is truly on-axis with cup.
                             # Positive = camera is left of shooter, negative = camera is right.
                             # Find empirically: fire a shot manually, note x_norm at that moment.
MIN_CONFIDENCE  = 0.35      # minimum detection confidence to attempt a shot

FLYWHEEL_THROTTLE  = 800    # DShot throttle while shooting (0-2047, tune empirically)
FLYWHEEL_SPINUP_S  = 2.0    # seconds to wait after SPIN command before feeding

# Empirical trajectory table: distance_m → (pitch_elevation_deg, dshot_throttle)
# Positive pitch = elevate upward.  Add entries as you calibrate.
TRAJECTORY_TABLE: dict[float, Tuple[float, int]] = {
    0.8: (3.0,  650),
    1.0: (5.0,  700),
    1.5: (8.0,  750),
    2.0: (12.0, 800),
    2.5: (15.0, 850),
    3.0: (18.0, 900),
    3.5: (21.0, 950),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_trajectory(distance_m: float) -> Tuple[float, int]:
    """
    Linear interpolation over TRAJECTORY_TABLE.
    Returns (pitch_deg, dshot_throttle). Clamps to table bounds.
    """
    keys = sorted(TRAJECTORY_TABLE.keys())
    if not keys:
        raise ValueError("TRAJECTORY_TABLE is empty — add calibration entries")

    if distance_m <= keys[0]:
        return TRAJECTORY_TABLE[keys[0]]
    if distance_m >= keys[-1]:
        return TRAJECTORY_TABLE[keys[-1]]

    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= distance_m <= hi:
            t = (distance_m - lo) / (hi - lo)
            p_lo, s_lo = TRAJECTORY_TABLE[lo]
            p_hi, s_hi = TRAJECTORY_TABLE[hi]
            pitch = p_lo + t * (p_hi - p_lo)
            throttle = int(round(s_lo + t * (s_hi - s_lo)))
            return pitch, throttle

    return TRAJECTORY_TABLE[keys[-1]]  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------------

class TurretController:
    def __init__(self, pico: PicoComms, detector: CupDetector) -> None:
        self.pico = pico
        self.detector = detector
        self._current_pitch_deg = 0.0  # tracked pitch position relative to home
        self._armed = False

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def home(self) -> bool:
        """
        Zero the Pico's position counters and reset internal state.
        Operator must have manually placed the turret at the home position
        (pitch fully down touching the base stop) before calling this.
        """
        ok = self.pico.home()
        if ok:
            self._current_pitch_deg = 0.0
            logger.info("Homed — counters zeroed")
        else:
            logger.error("HOME command failed")
        return ok

    def arm(self, throttle: int = FLYWHEEL_THROTTLE) -> bool:
        """Spin up flywheels and wait for them to reach speed."""
        logger.info("Spinning up flywheels (throttle=%d)…", throttle)
        ok = self.pico.set_spin(throttle, throttle)
        if not ok:
            logger.error("SPIN command failed")
            return False
        time.sleep(FLYWHEEL_SPINUP_S)
        self._armed = True
        logger.info("Flywheels armed")
        return True

    def disarm(self) -> bool:
        """Stop flywheels."""
        self._armed = False
        ok = self.pico.set_spin(0, 0)
        if ok:
            logger.info("Flywheels disarmed")
        return ok

    def estop(self) -> None:
        """Emergency stop — call from any thread."""
        self._armed = False
        self.pico.estop()
        logger.warning("ESTOP triggered")

    # ------------------------------------------------------------------
    # Aiming
    # ------------------------------------------------------------------

    def align_yaw(
        self,
        max_iterations: int = 10,
        timeout_per_move: float = 4.0,
        settle_s: float = 0.15,
    ) -> bool:
        """
        Iteratively move yaw until the detected cup is horizontally centered
        (|x_norm| < YAW_DEAD_ZONE).

        Since the camera is mounted on the yaw axis, each correction move
        directly re-centers the view.

        Returns True if converged, False if couldn't lock on or timed out.
        """
        for iteration in range(max_iterations):
            result = self.detector.get_result()

            if not result.valid or result.confidence < MIN_CONFIDENCE:
                logger.warning("align_yaw iter %d: no valid target", iteration)
                return False

            x_norm = result.x_norm
            logger.debug("align_yaw iter %d: x_norm=%.4f", iteration, x_norm)

            error = x_norm - AIM_X_OFFSET
            if abs(error) < YAW_DEAD_ZONE:
                logger.info("Yaw aligned after %d iterations (x_norm=%.4f, offset=%.4f)",
                            iteration, x_norm, AIM_X_OFFSET)
                return True

            # Map pixel error to angle.
            # error = +1.0 means cup is one half-frame to the right of the aim point.
            delta_deg = error * (CAMERA_HFOV_DEG / 2.0)
            logger.info("align_yaw iter %d: moving %.2f°", iteration, delta_deg)

            if not self.pico.move_yaw_sync(delta_deg, YAW_ALIGN_FREQ_HZ, timeout_per_move):
                logger.error("Yaw move timed out or failed on iteration %d", iteration)
                return False

            # Brief pause for camera to capture a fresh frame after motion
            time.sleep(settle_s)

        logger.warning("Yaw alignment did not converge in %d iterations", max_iterations)
        return False

    def set_pitch_for_distance(self, distance_m: float, timeout: float = 8.0) -> bool:
        """
        Look up pitch elevation angle for the given distance, compute delta
        from current position, and send the PITCH command.
        """
        target_pitch, throttle = interpolate_trajectory(distance_m)
        delta_deg = target_pitch - self._current_pitch_deg

        logger.info(
            "set_pitch: dist=%.2fm → target_pitch=%.1f° (delta=%.1f°) throttle=%d",
            distance_m, target_pitch, delta_deg, throttle,
        )

        # Update flywheel speed to match distance if armed
        if self._armed:
            self.pico.set_spin(throttle, throttle)

        if abs(delta_deg) < 0.5:
            logger.debug("Pitch delta too small (%.2f°), skipping move", delta_deg)
            return True

        ok = self.pico.move_pitch_sync(delta_deg, PITCH_FREQ_HZ, timeout)
        if ok:
            self._current_pitch_deg = target_pitch
        else:
            logger.error("Pitch move failed or timed out")
        return ok

    # ------------------------------------------------------------------
    # Fire sequence
    # ------------------------------------------------------------------

    def fire_sequence(self) -> bool:
        """
        Full single-shot sequence:
          1. Check we have a confident detection
          2. Align yaw
          3. Re-detect to get fresh distance estimate
          4. Set pitch for distance
          5. Fire (feed motor)
        Returns True if a shot was dispatched.
        """
        if not self._armed:
            logger.warning("fire_sequence: not armed — call arm() first")
            return False

        # Initial detection check
        result = self.detector.get_result()
        if not result.valid or result.confidence < MIN_CONFIDENCE:
            logger.warning("fire_sequence: no valid target (conf=%.2f)", result.confidence)
            return False

        # Align yaw
        if not self.align_yaw():
            logger.warning("fire_sequence: yaw alignment failed")
            return False

        # Fresh detection after alignment
        time.sleep(0.1)
        result = self.detector.get_result()
        if not result.valid:
            logger.warning("fire_sequence: lost target after yaw alignment")
            return False

        distance_m = result.distance_m
        if distance_m <= 0:
            logger.warning("fire_sequence: invalid distance estimate %.2fm", distance_m)
            return False

        logger.info("fire_sequence: distance=%.2fm", distance_m)

        # Set pitch
        if not self.set_pitch_for_distance(distance_m):
            logger.warning("fire_sequence: pitch set failed")
            return False

        # Fire
        logger.info("Firing!")
        ok = self.pico.feed_sync(timeout=5.0)
        if not ok:
            logger.error("Feed timed out or failed")
        return ok

    # ------------------------------------------------------------------
    # Auto loop
    # ------------------------------------------------------------------

    def run_once(self) -> bool:
        """Attempt one detect → aim → fire cycle."""
        return self.fire_sequence()

    def run_auto(
        self,
        shots: int = 10,
        delay_between_shots: float = 1.5,
    ) -> None:
        """
        Run the auto loop for <shots> attempts.
        Calls disarm() when finished.
        """
        logger.info("Starting auto loop: %d shots, %.1fs between shots", shots, delay_between_shots)
        hits = 0
        for i in range(shots):
            logger.info("--- Shot %d / %d ---", i + 1, shots)
            ok = self.run_once()
            if ok:
                hits += 1
            time.sleep(delay_between_shots)

        logger.info("Auto loop complete: %d / %d shots fired", hits, shots)
        self.disarm()


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_auto(args: argparse.Namespace) -> None:
    pico = PicoComms(port=args.port)
    detector = CupDetector(camera_index=args.camera, focal_length_px=args.focal)

    if not pico.connect():
        logger.error("Could not connect to Pico on %s", args.port)
        sys.exit(1)

    detector.start()
    time.sleep(0.5)  # let camera settle

    ctrl = TurretController(pico, detector)
    try:
        ctrl.home()
        ctrl.arm(throttle=args.throttle)
        ctrl.run_auto(shots=args.shots, delay_between_shots=args.delay)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        ctrl.estop()
    finally:
        detector.stop()
        pico.disconnect()


def _run_manual(args: argparse.Namespace) -> None:
    """Interactive shell for manual turret control."""
    pico = PicoComms(port=args.port)
    detector = CupDetector(camera_index=args.camera, focal_length_px=args.focal)

    if not pico.connect():
        logger.error("Could not connect to Pico on %s\n"
                     "  Check: ls /dev/ttyACM* to find the correct port.", args.port)
        sys.exit(1)

    detector.start()
    ctrl = TurretController(pico, detector)

    print("Manual control shell. Commands:")
    print("  home          — zero counters")
    print("  arm [t]       — spin up flywheels (default throttle=800)")
    print("  disarm        — stop flywheels")
    print("  yaw <deg>     — move yaw")
    print("  pitch <deg>   — move pitch")
    print("  align         — auto-align yaw to cup")
    print("  fire          — full fire sequence")
    print("  status        — query Pico status")
    print("  detect        — print latest detection")
    print("  estop         — emergency stop")
    print("  quit          — exit")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "quit":
                break
            elif cmd == "home":
                ctrl.home()
            elif cmd == "arm":
                t = int(parts[1]) if len(parts) > 1 else FLYWHEEL_THROTTLE
                ctrl.arm(t)
            elif cmd == "disarm":
                ctrl.disarm()
            elif cmd == "yaw":
                if len(parts) < 2:
                    print("Usage: yaw <degrees>")
                else:
                    pico.move_yaw_sync(float(parts[1]))
            elif cmd == "pitch":
                if len(parts) < 2:
                    print("Usage: pitch <degrees>")
                else:
                    pico.move_pitch_sync(float(parts[1]))
            elif cmd == "align":
                ok = ctrl.align_yaw()
                print("Aligned" if ok else "Failed to align")
            elif cmd == "fire":
                ok = ctrl.fire_sequence()
                print("Shot fired" if ok else "Fire sequence failed")
            elif cmd == "status":
                s = pico.get_status()
                print(f"yaw_moving={s.yaw_moving} pitch_moving={s.pitch_moving} "
                      f"feed={s.feed_active} spin1={s.spin1} spin2={s.spin2}")
            elif cmd == "detect":
                r = detector.get_result()
                print(f"valid={r.valid} x={r.x_norm:.3f} y={r.y_norm:.3f} "
                      f"dist={r.distance_m:.2f}m conf={r.confidence:.2f}")
            elif cmd == "estop":
                ctrl.estop()
            else:
                print(f"Unknown command: {cmd}")

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.estop()
        detector.stop()
        pico.disconnect()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Beer Pong Turret Controller")
    parser.add_argument("--port",     default="/dev/ttyACM0", help="Pico serial port")
    parser.add_argument("--camera",   type=int,   default=0,    help="Camera index")
    parser.add_argument("--focal",    type=float, default=700.0, help="Camera focal length (px)")
    parser.add_argument("--shots",    type=int,   default=6,    help="Number of shots in auto mode")
    parser.add_argument("--delay",    type=float, default=1.5,  help="Delay between shots (s)")
    parser.add_argument("--throttle", type=int,   default=FLYWHEEL_THROTTLE, help="Flywheel DShot throttle")
    parser.add_argument("--manual",   action="store_true", help="Interactive manual control mode")
    args = parser.parse_args()

    if args.manual:
        _run_manual(args)
    else:
        _run_auto(args)


if __name__ == "__main__":
    main()
