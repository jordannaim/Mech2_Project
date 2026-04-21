"""
controller.py — Turret orchestration: detect → align → aim → fire.

Trajectory lookup table maps measured distance to pitch step count (from home)
and flywheel throttle.  Fill this in empirically: set pitch from home at each
test distance until balls land in the cup, then add entries to TRAJECTORY_TABLE.

Usage:
    python -m Integration.controller            # auto loop
    python -m Integration.controller --manual   # interactive shell
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from typing import Tuple

import cv2

from .pico_comms import PicoComms
from .vision import CupDetector, DetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical / tuning constants
# ---------------------------------------------------------------------------

CAMERA_HFOV_DEG = 60.0
CAMERA_VFOV_DEG = 40.0

YAW_VEL_KP             = 24
YAW_VEL_KI             = 0.6
YAW_VEL_MIN_HZ         = 35
YAW_VEL_MAX_HZ         = 200
YAW_ALIGN_LOOP_S       = 0.05
YAW_VEL_ACCEL_HZ_PER_S = 450
YAW_VEL_DECEL_HZ_PER_S = 1400
YAW_ERROR_ALPHA        = 0.50
YAW_I_ACTIVE_BAND      = 0.35
YAW_I_MAX              = 0.25
YAW_STARTUP_FRAMES     = 4
YAW_STARTUP_MAX_HZ     = 45
YAW_NEAR_CENTER_BAND   = 0.05
YAW_NEAR_CENTER_MIN_HZ = 3
YAW_BRAKE_BAND         = 0.12
YAW_BRAKE_MAX_HZ       = 10
YAW_STOP_BAND          = 0.15
YAW_CROSS_DAMP         = 0.55
YAW_LOOKAHEAD_S        = 0.12
YAW_FRAME_STALE_S      = 0.14
YAW_PRED_MIX           = 0.60
YAW_SETTLE_VERIFY_S    = 0.20
YAW_SETTLE_VERIFY_FRAMES = 3
YAW_LOST_GRACE_FRAMES  = 3
YAW_EDGE_ABORT_NORM    = 0.92
YAW_COARSE_FREQ_HZ     = 600
PITCH_FREQ_HZ          = 400

# Step angles (MS1=MS2=HIGH → 16 microsteps on all axes)
YAW_STEP_ANGLE_DEG   = 1.8 / 16        # 0.1125° per yaw microstep
PITCH_STEP_ANGLE_DEG = 1.8 / 16 / 10   # 0.01125° per pitch microstep (output shaft, 10:1 gearbox)
STEPPER_STEP_ANGLE_DEG = YAW_STEP_ANGLE_DEG  # used in yaw alignment math

YAW_LOST_SEARCH_S      = 1.25

PITCH_ENABLE_MOTION          = True   # pitch hardware is now correctly wired
PITCH_ALLOW_FIRE_WITHOUT_MOVE = True  # still fire if pitch move fails

YAW_DEAD_ZONE    = 0.04
MIN_CONFIDENCE   = 0.7
AIM_OFFSET_SCALE = 1
AIM_OFFSET_MAX   = 0.00

CAMERA_X_OFFSET_M = 0

FLYWHEEL_THROTTLE_1 = 200
FLYWHEEL_THROTTLE_2 = 200
FLYWHEEL_SPINUP_S   = 3.0

# Indexer stepper configuration
INDEXER_STEPS_PER_BALL = 800   # steps to index one ball — CALIBRATE EMPIRICALLY
INDEXER_FREQ_HZ        = 400   # step frequency for indexer

# Empirical trajectory table: distance_m → (pitch_steps_from_home, dshot_throttle_1, dshot_throttle_2)
# pitch_steps_from_home is the ABSOLUTE step count from the home (fully-leaned-back) position.
# Positive = pitched up.  All entries are PLACEHOLDERS — calibrate before use.
TRAJECTORY_TABLE: dict[float, Tuple[int, int, int]] = {
    1.4: (2000,  188, 246),
    1.6: (4000,  190, 255),
    1.8: (7000,  195, 260),
    1.9: (9000,  205, 265),
    2.0: (11000, 210, 270),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_trajectory(distance_m: float) -> Tuple[int, int, int]:
    """
    Linear interpolation over TRAJECTORY_TABLE.
    Returns (pitch_steps_from_home, dshot_throttle_1, dshot_throttle_2).
    Clamps to table bounds.
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
            s_lo, s1_lo, s2_lo = TRAJECTORY_TABLE[lo]
            s_hi, s1_hi, s2_hi = TRAJECTORY_TABLE[hi]
            pitch_steps = int(round(s_lo + t * (s_hi - s_lo)))
            throttle1   = int(round(s1_lo + t * (s1_hi - s1_lo)))
            throttle2   = int(round(s2_lo + t * (s2_hi - s2_lo)))
            return pitch_steps, throttle1, throttle2

    return TRAJECTORY_TABLE[keys[-1]]


def _compute_aim_offset(distance_m: float) -> float:
    if distance_m <= 0.0:
        return 0.0
    theta = math.atan(CAMERA_X_OFFSET_M / distance_m)
    hfov_rad = math.radians(CAMERA_HFOV_DEG)
    raw = -theta / (hfov_rad / 2.0)
    return float(max(-AIM_OFFSET_MAX, min(AIM_OFFSET_MAX, raw * AIM_OFFSET_SCALE)))


# ---------------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------------

class TurretController:
    def __init__(self, pico: PicoComms, detector: CupDetector) -> None:
        self.pico = pico
        self.detector = detector
        self._current_pitch_steps: int = 0  # absolute step count from home (0 = fully leaned back)
        self._armed = False

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def home(self) -> bool:
        """
        Zero Pico counters and reset pitch tracking.
        Operator must physically place turret at the home position
        (pitch fully leaned back against the back-stop) before calling this.
        """
        ok = self.pico.home()
        if ok:
            self._current_pitch_steps = 0
            logger.info("Homed — pitch_steps reset to 0")
        else:
            logger.error("HOME command failed")
        return ok

    def arm(
        self,
        throttle1: int = FLYWHEEL_THROTTLE_1,
        throttle2: int = FLYWHEEL_THROTTLE_2,
    ) -> bool:
        logger.info("Spinning up flywheels (t1=%d t2=%d)…", throttle1, throttle2)
        ok = self.pico.set_spin(throttle1, throttle2)
        if not ok:
            logger.error("SPIN command failed")
            return False
        time.sleep(FLYWHEEL_SPINUP_S)
        self._armed = True
        logger.info("Flywheels armed")
        return True

    def disarm(self) -> bool:
        self._armed = False
        ok = self.pico.set_spin(0, 0)
        if ok:
            logger.info("Flywheels disarmed")
        return ok

    def estop(self) -> None:
        self._armed = False
        self.pico.estop()
        logger.warning("ESTOP triggered")

    # ------------------------------------------------------------------
    # Aiming
    # ------------------------------------------------------------------

    def align_yaw(self, timeout: float = 8.0) -> bool:
        """
        Stepwise predictive align: move a small chunk, re-detect, shrink the
        chunk as the target gets closer to center, declare success after the
        centered cup stays centered for a short settle window.
        """
        deadline = time.monotonic() + timeout
        locked_here = False

        if not self.detector.is_target_locked():
            seed = self.detector.get_result()
            if not seed.valid or seed.confidence < MIN_CONFIDENCE:
                logger.warning("align_yaw: no valid target to lock")
                return False
            self.detector.lock_target(seed.x_norm, seed.y_norm)
            locked_here = True

        chunk_steps = 1
        chunk_deg = chunk_steps * STEPPER_STEP_ANGLE_DEG
        burst_freq_hz = 20
        burst_pulses = 1
        post_align_backstep_bursts = 2
        settle_frames_needed = 3
        settle_timeout_s = 0.20

        try:
            settle_frames = 0
            last_direction = 0

            while time.monotonic() < deadline:
                result = self.detector.get_result()

                if not result.valid or result.confidence < MIN_CONFIDENCE:
                    time.sleep(YAW_ALIGN_LOOP_S)
                    continue

                aim_offset = _compute_aim_offset(result.distance_m)
                error = result.x_norm - aim_offset

                if abs(error) < YAW_DEAD_ZONE:
                    settle_frames += 1
                    if settle_frames < settle_frames_needed:
                        time.sleep(YAW_ALIGN_LOOP_S)
                        continue

                    settle_deadline = time.monotonic() + settle_timeout_s
                    stable_count = 0
                    while time.monotonic() < settle_deadline:
                        time.sleep(YAW_ALIGN_LOOP_S)
                        verify = self.detector.get_result()
                        if not verify.valid or verify.confidence < MIN_CONFIDENCE:
                            stable_count = 0
                            continue
                        verify_error = verify.x_norm - _compute_aim_offset(verify.distance_m)
                        if abs(verify_error) < YAW_DEAD_ZONE:
                            stable_count += 1
                            if stable_count >= settle_frames_needed:
                                if post_align_backstep_bursts > 0 and last_direction != 0:
                                    reverse_hz = -last_direction * burst_freq_hz
                                    reverse_duration_s = max(0.018, min(0.030, burst_pulses / float(burst_freq_hz)))
                                    for _ in range(post_align_backstep_bursts):
                                        if not self.pico.set_yaw_velocity(reverse_hz, ack_timeout=0.15):
                                            logger.warning("align_yaw: failed post-align reverse burst")
                                            break
                                        time.sleep(reverse_duration_s)
                                        self.pico.set_yaw_velocity(0, ack_timeout=0.05)
                                        time.sleep(YAW_ALIGN_LOOP_S)
                                logger.info(
                                    "Yaw aligned (x_norm=%.4f error=%.4f)",
                                    verify.x_norm, verify_error,
                                )
                                return True
                        else:
                            stable_count = 0

                    settle_frames = 0
                    last_direction = 0
                    continue

                settle_frames = 0

                error_deg = error * (CAMERA_HFOV_DEG / 2.0)
                abs_error_deg = abs(error_deg)
                direction = 1 if error_deg > 0 else -1

                if last_direction != 0 and direction != last_direction and abs_error_deg <= 1.0:
                    self.pico.set_yaw_velocity(0)
                    time.sleep(YAW_ALIGN_LOOP_S)
                    last_direction = direction
                    continue

                last_direction = direction

                cmd_deg = chunk_deg * direction
                burst_duration_s = max(0.018, min(0.030, burst_pulses / float(burst_freq_hz)))
                burst_hz = burst_freq_hz * direction

                logger.debug(
                    "align_yaw burst: error_deg=%.3f cmd_deg=%.3f burst_hz=%d duration=%.3f",
                    error_deg, cmd_deg, burst_hz, burst_duration_s,
                )

                if not self.pico.set_yaw_velocity(burst_hz, ack_timeout=0.15):
                    logger.warning("align_yaw: failed to start yaw burst")
                    return False

                time.sleep(burst_duration_s)
                self.pico.set_yaw_velocity(0, ack_timeout=0.05)
                time.sleep(YAW_ALIGN_LOOP_S)

            logger.warning("align_yaw: timed out after %.1fs", timeout)
            return False
        finally:
            if locked_here:
                self.detector.unlock_target()

    def set_pitch_for_distance(self, distance_m: float, timeout: float = 15.0) -> bool:
        """
        Look up the pitch step target for the given distance and move the
        pitch stepper to that absolute step count from home.
        """
        target_steps, throttle1, throttle2 = interpolate_trajectory(distance_m)
        delta_steps = target_steps - self._current_pitch_steps

        logger.info(
            "set_pitch: dist=%.2fm → target_steps=%d (current=%d, delta=%d) t1=%d t2=%d",
            distance_m, target_steps, self._current_pitch_steps, delta_steps,
            throttle1, throttle2,
        )

        if self._armed:
            self.pico.set_spin(throttle1, throttle2)

        if not PITCH_ENABLE_MOTION:
            logger.warning("set_pitch: PITCH_ENABLE_MOTION=False — skipping move")
            return True

        if delta_steps == 0:
            logger.debug("Pitch delta is 0 steps — no move needed")
            return True

        ok = self.pico.move_pitch_steps_sync(delta_steps, PITCH_FREQ_HZ, timeout)
        if ok:
            self._current_pitch_steps = target_steps
            logger.info("Pitch moved to %d steps from home", self._current_pitch_steps)
            return True

        logger.error("Pitch move failed or timed out")
        return False

    # ------------------------------------------------------------------
    # Fire sequence
    # ------------------------------------------------------------------

    def fire_sequence(self) -> bool:
        """
        Full single-shot sequence:
          1. Acquire confident detection
          2. Align yaw
          3. Re-detect for fresh distance
          4. Set pitch for distance
          5. Fire: indexer stepper + servo extend/retract
        Returns True if a shot was dispatched.
        """
        if not self._armed:
            logger.warning("fire_sequence: not armed — call arm() first")
            return False

        result = DetectionResult(valid=False)
        acquire_deadline = time.monotonic() + 0.8
        while time.monotonic() < acquire_deadline:
            result = self.detector.get_result()
            if result.valid and result.confidence >= MIN_CONFIDENCE:
                break
            time.sleep(0.05)

        if not result.valid or result.confidence < MIN_CONFIDENCE:
            logger.warning("fire_sequence: no valid target (conf=%.2f)", result.confidence)
            return False

        self.detector.lock_target(result.x_norm, result.y_norm)
        logger.info(
            "fire_sequence: target x=%.3f y=%.3f conf=%.2f",
            result.x_norm, result.y_norm, result.confidence,
        )

        try:
            if not self.align_yaw():
                logger.warning("fire_sequence: yaw alignment failed")
                return False

            result = DetectionResult(valid=False)
            settle_deadline = time.monotonic() + 0.75
            while time.monotonic() < settle_deadline:
                time.sleep(0.05)
                result = self.detector.get_result()
                if result.valid and result.confidence >= MIN_CONFIDENCE:
                    break

            if not result.valid or result.confidence < MIN_CONFIDENCE:
                logger.warning("fire_sequence: lost target after yaw alignment")
                return False

            distance_m = result.distance_m
            if distance_m <= 0:
                logger.warning("fire_sequence: invalid distance %.2fm", distance_m)
                return False

            logger.info("fire_sequence: distance=%.2fm", distance_m)

            if not self.set_pitch_for_distance(distance_m):
                if PITCH_ALLOW_FIRE_WITHOUT_MOVE:
                    logger.warning("fire_sequence: pitch failed — firing anyway")
                else:
                    return False

            # Fire: indexer moves ball to position, servo pushes it into flywheels
            logger.info("Firing! indexer_steps=%d", INDEXER_STEPS_PER_BALL)
            ok = self.pico.fire_sync(INDEXER_STEPS_PER_BALL, timeout=8.0)
            if not ok:
                logger.error("Fire sequence timed out or failed")
            return ok
        finally:
            self.detector.unlock_target()

    # ------------------------------------------------------------------
    # Auto loop
    # ------------------------------------------------------------------

    def run_once(self) -> bool:
        return self.fire_sequence()

    def run_auto(self, shots: int = 10, delay_between_shots: float = 1.5) -> None:
        logger.info("Auto loop: %d shots, %.1fs between", shots, delay_between_shots)
        hits = 0
        for i in range(shots):
            logger.info("--- Shot %d / %d ---", i + 1, shots)
            ok = self.run_once()
            if ok:
                hits += 1
            time.sleep(delay_between_shots)
        logger.info("Auto loop done: %d / %d fired", hits, shots)
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
    time.sleep(0.5)

    ctrl = TurretController(pico, detector)
    try:
        ctrl.home()
        ctrl.arm(throttle1=args.throttle1, throttle2=args.throttle2)
        ctrl.run_auto(shots=args.shots, delay_between_shots=args.delay)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        ctrl.estop()
    finally:
        detector.stop()
        pico.disconnect()


def _run_manual(args: argparse.Namespace) -> None:
    pico = PicoComms(port=args.port)
    detector = CupDetector(camera_index=args.camera, focal_length_px=args.focal)

    if not pico.connect():
        logger.error("Could not connect to Pico on %s\n"
                     "  Check: ls /dev/ttyACM* to find the correct port.", args.port)
        sys.exit(1)

    detector.start()
    ctrl = TurretController(pico, detector)

    _display_stop = threading.Event()

    def _display_loop() -> None:
        while not _display_stop.is_set():
            frame = detector.get_annotated_frame()
            if frame is not None:
                cv2.imshow("Cup Detector", frame)
            cv2.waitKey(1)

    display_thread = threading.Thread(target=_display_loop, daemon=True, name="display")
    display_thread.start()

    print("Manual control shell.  Commands:")
    print("  home                — zero pitch counter (turret must be at back-stop)")
    print("  arm [t1] [t2]       — spin up flywheels")
    print("  disarm              — stop flywheels")
    print("  yaw <deg>           — move yaw by degrees")
    print("  pitch <steps>       — move pitch by raw step count (signed)")
    print("  indexer <steps>     — test indexer + servo fire (e.g. indexer 800)")
    print("  feed                — fire sequence with default indexer steps")
    print("  align               — auto-align yaw to cup")
    print("  fire                — full fire sequence")
    print("  status              — query Pico status")
    print("  detect              — print latest detection")
    print("  estop               — emergency stop")
    print("  quit                — exit")

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
                if len(parts) > 2:
                    t1, t2 = int(parts[1]), int(parts[2])
                elif len(parts) > 1:
                    t1 = t2 = int(parts[1])
                else:
                    t1, t2 = FLYWHEEL_THROTTLE_1, FLYWHEEL_THROTTLE_2
                ctrl.arm(t1, t2)
            elif cmd == "disarm":
                ctrl.disarm()
            elif cmd == "yaw":
                if len(parts) < 2:
                    print("Usage: yaw <degrees>")
                else:
                    pico.move_yaw_sync(float(parts[1]))
            elif cmd == "pitch":
                if len(parts) < 2:
                    print("Usage: pitch <steps>  (raw signed step count)")
                else:
                    steps = int(parts[1])
                    ok = pico.move_pitch_steps_sync(steps, PITCH_FREQ_HZ)
                    if ok:
                        ctrl._current_pitch_steps += steps
                        print(f"Pitch now at {ctrl._current_pitch_steps} steps from home")
                    else:
                        print("Pitch move failed")
            elif cmd == "indexer":
                if len(parts) < 2:
                    print("Usage: indexer <steps>")
                else:
                    steps = int(parts[1])
                    print(f"Firing indexer {steps} steps + servo…")
                    ok = pico.fire_sync(steps, timeout=8.0)
                    print("Done" if ok else "Failed")
            elif cmd == "feed":
                print(f"Firing with default {INDEXER_STEPS_PER_BALL} indexer steps + servo…")
                ok = pico.fire_sync(INDEXER_STEPS_PER_BALL, timeout=8.0)
                print("Done" if ok else "Failed")
            elif cmd == "align":
                ok = ctrl.align_yaw()
                print("Aligned" if ok else "Failed to align")
            elif cmd == "fire":
                ok = ctrl.fire_sequence()
                print("Shot fired" if ok else "Fire sequence failed")
            elif cmd == "status":
                s = pico.get_status()
                print(f"yaw_moving={s.yaw_moving} pitch_moving={s.pitch_moving} "
                      f"indexer_moving={s.indexer_moving} fire_active={s.fire_active} "
                      f"spin1={s.spin1} spin2={s.spin2}")
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
        _display_stop.set()
        cv2.destroyAllWindows()
        ctrl.estop()
        detector.stop()
        pico.disconnect()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Beer Pong Turret Controller")
    parser.add_argument("--port",      default="/dev/ttyACM0")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--focal",     type=float, default=1050.0)
    parser.add_argument("--shots",     type=int,   default=6)
    parser.add_argument("--delay",     type=float, default=1.5)
    parser.add_argument("--throttle1", type=int,   default=FLYWHEEL_THROTTLE_1)
    parser.add_argument("--throttle2", type=int,   default=FLYWHEEL_THROTTLE_2)
    parser.add_argument("--manual",    action="store_true")
    args = parser.parse_args()

    if args.manual:
        _run_manual(args)
    else:
        _run_auto(args)


if __name__ == "__main__":
    main()
