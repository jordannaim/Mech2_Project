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
# Physical / tuning constants  (adjust these as you calibrate)
# ---------------------------------------------------------------------------

CAMERA_HFOV_DEG = 60.0      # horizontal field of view of the camera
CAMERA_VFOV_DEG = 40.0      # vertical field of view

YAW_VEL_KP             = 24  # proportional gain: maps |error| (0-1) to step freq (Hz)
YAW_VEL_KI             = 0.6 # integral gain: trims residual bias and steady-state error
YAW_VEL_MIN_HZ         = 35     # baseline minimum frequency sent
YAW_VEL_MAX_HZ         = 200    # cap max frequency to reduce camera shake [REDUCED]
YAW_ALIGN_LOOP_S       = 0.05   # control loop period (20 Hz) — must be < one camera frame
YAW_VEL_ACCEL_HZ_PER_S = 450    # gentler acceleration/deceleration for smoother motion
YAW_VEL_DECEL_HZ_PER_S = 1400   # brake faster than accelerate to stop on target
YAW_ERROR_ALPHA        = 0.50   # EMA smoothing for x-error (lower = smoother) [INCREASED]
YAW_I_ACTIVE_BAND      = 0.35   # only integrate when |error| is inside this band (anti-windup)
YAW_I_MAX              = 0.25   # clamp integral state (error*s)
YAW_STARTUP_FRAMES     = 4      # first valid frames use conservative speed cap
YAW_STARTUP_MAX_HZ     = 45     # startup cap so first movement is confident, not a jump
YAW_NEAR_CENTER_BAND   = 0.05   # use gentler minimum speed near center to avoid ping-pong
YAW_NEAR_CENTER_MIN_HZ = 3      # very slow final creep near center
YAW_BRAKE_BAND         = 0.12   # begin explicit braking earlier to stop before center
YAW_BRAKE_MAX_HZ       = 10     # cap speed in brake band to avoid sailing past center
YAW_STOP_BAND          = 0.15   # if prediction says we'll cross within this band, stop now
YAW_CROSS_DAMP         = 0.55   # damp command right after crossing centerline
YAW_LOOKAHEAD_S        = 0.12   # predictive lead time to compensate vision/control latency
YAW_FRAME_STALE_S      = 0.14   # hold motion if newest frame is older than this
YAW_PRED_MIX           = 0.60   # blend predicted error into control error (0..1)
YAW_SETTLE_VERIFY_S    = 0.20   # verify the turret stays centered after stopping
YAW_SETTLE_VERIFY_FRAMES = 3    # consecutive centered frames required after stop
YAW_LOST_GRACE_FRAMES  = 3      # tolerate brief vision dropouts before aborting align
YAW_EDGE_ABORT_NORM    = 0.92   # stop if target reaches frame edge and command pushes outward
YAW_COARSE_FREQ_HZ = 600    # step freq for large non-vision moves
PITCH_FREQ_HZ      = 400    # step freq for pitch moves
STEPPER_STEP_ANGLE_DEG = 1.8/8  # must match Pico firmware step angle
YAW_LOST_SEARCH_S       = 1.25   # keep searching through temporary vision dropouts

# Pitch debug/fallback behavior while uncalibrated or on mixed firmware.
PITCH_ENABLE_MOTION = False           # keep False until pitch is calibrated
PITCH_ALLOW_FIRE_WITHOUT_MOVE = True  # still allow feed/fire when pitch move is skipped/fails
PITCH_LEGACY_STEP_ANGLE_DEG = 1.8     # fallback if Pico is still on full-step firmware
PITCH_FALLBACK_FREQ_HZ = 250          # slower retry frequency for fallback moves

YAW_DEAD_ZONE   = 0.04     # |x_norm - aim_offset| below this = centered enough to fire
MIN_CONFIDENCE  = 0.7       # minimum detection confidence to attempt a shot
AIM_OFFSET_SCALE = 1     # scale camera-to-shooter x offset compensation [REDUCED]
AIM_OFFSET_MAX   = 0.00     # clamp |aim offset| so alignment doesn't run off-frame

# Camera is mounted 2.2 inches (0.0559 m) to the RIGHT of the shooter centerline.
# The aim offset in x_norm is distance-dependent: use _compute_aim_offset(distance_m).
CAMERA_X_OFFSET_M = 0

FLYWHEEL_THROTTLE_1 = 200   # DShot throttle for flywheel 1 (0-2047)
FLYWHEEL_THROTTLE_2 = 200   # DShot throttle for flywheel 2 (0-2047)
FLYWHEEL_SPINUP_S  = 3.0    # seconds to wait after SPIN command before feeding
FEED_ON_TIME_S     = 0.75    # feed motor on-time per shot (seconds)

# Empirical trajectory table: distance_m → (pitch_elevation_deg, dshot_throttle_1, dshot_throttle_2)
# Positive pitch = elevate upward.  Add entries as you calibrate.
TRAJECTORY_TABLE: dict[float, Tuple[float, int, int]] = {
    1.4: (5.0,  188, 246),
    1.6: (8.0,  190, 255),
    1.8: (12.0, 195, 260),
    1.9: (15.0, 205, 265),
    2.0: (18.0, 210, 270),
    
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_trajectory(distance_m: float) -> Tuple[float, int, int]:
    """
    Linear interpolation over TRAJECTORY_TABLE.
    Returns (pitch_deg, dshot_throttle_1, dshot_throttle_2). Clamps to table bounds.
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
            p_lo, s1_lo, s2_lo = TRAJECTORY_TABLE[lo]
            p_hi, s1_hi, s2_hi = TRAJECTORY_TABLE[hi]
            pitch = p_lo + t * (p_hi - p_lo)
            throttle1 = int(round(s1_lo + t * (s1_hi - s1_lo)))
            throttle2 = int(round(s2_lo + t * (s2_hi - s2_lo)))
            return pitch, throttle1, throttle2

    return TRAJECTORY_TABLE[keys[-1]]  # unreachable but satisfies type checker


def _compute_aim_offset(distance_m: float) -> float:
    """
    Return the x_norm value the cup should sit at when the shooter is truly
    on-axis, accounting for the lateral camera-to-shooter offset.

    When the camera is to the RIGHT of the shooter, the cup must appear
    slightly LEFT of camera center (negative x_norm) for the shooter to be
    pointed directly at it.

    Formula: θ = arctan(CAMERA_X_OFFSET_M / distance_m)
             x_norm_target = -θ / (HFOV_rad / 2)
    """
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

    def arm(
        self,
        throttle1: int = FLYWHEEL_THROTTLE_1,
        throttle2: int = FLYWHEEL_THROTTLE_2,
    ) -> bool:
        """Spin up flywheels and wait for them to reach speed."""
        logger.info("Spinning up flywheels (throttle1=%d throttle2=%d)…", throttle1, throttle2)
        ok = self.pico.set_spin(throttle1, throttle2)
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

    def align_yaw(self, timeout: float = 8.0) -> bool:
        """
        Stepwise predictive align: move a small chunk, re-detect, shrink the
        chunk as the target gets closer to center, and only declare success
        after the centered cup stays centered for a short settle window.
        """
        deadline = time.monotonic() + timeout
        locked_here = False

        # For manual 'align', lock a specific cup first so target identity does
        # not switch frame-to-frame.
        if not self.detector.is_target_locked():
            seed = self.detector.get_result()
            if not seed.valid or seed.confidence < MIN_CONFIDENCE:
                logger.warning("align_yaw: no valid target to lock")
                return False
            self.detector.lock_target(seed.x_norm, seed.y_norm)
            locked_here = True

        # Debug mode: try the smallest practical correction first using a
        # timed velocity burst, then re-detect after each burst.
        chunk_steps = 1
        chunk_deg = chunk_steps * STEPPER_STEP_ANGLE_DEG
        burst_freq_hz = 20
        burst_pulses = 1
        post_align_backstep_bursts = 2  # send this many reverse bursts after align, then stop
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
                                # Optional post-align correction: move back one burst
                                # opposite the last approach direction, then stop.
                                if post_align_backstep_bursts > 0 and last_direction != 0:
                                    reverse_hz = -last_direction * burst_freq_hz
                                    reverse_duration_s = max(0.018, min(0.030, burst_pulses / float(burst_freq_hz)))
                                    for _ in range(post_align_backstep_bursts):
                                        if not self.pico.set_yaw_velocity(reverse_hz, ack_timeout=0.15):
                                            logger.warning("align_yaw: failed to apply post-align reverse burst")
                                            break
                                        time.sleep(reverse_duration_s)
                                        self.pico.set_yaw_velocity(0, ack_timeout=0.05)
                                        time.sleep(YAW_ALIGN_LOOP_S)
                                logger.info(
                                    "Yaw aligned (x_norm=%.4f error=%.4f)",
                                    verify.x_norm,
                                    verify_error,
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

                # If we crossed the center since the last step, stop and let
                # the detector measure the new position before doing anything else.
                if last_direction != 0 and direction != last_direction and abs_error_deg <= 1.0:
                    self.pico.set_yaw_velocity(0)
                    time.sleep(YAW_ALIGN_LOOP_S)
                    last_direction = direction
                    continue

                last_direction = direction

                cmd_deg = chunk_deg * direction
                logger.debug(
                    "align_yaw step: error_deg=%.3f cmd_deg=%.3f",
                    error_deg,
                    cmd_deg,
                )

                # One pulse at the chosen frequency is the smallest meaningful
                # motion in velocity mode.
                burst_duration_s = max(0.018, min(0.030, burst_pulses / float(burst_freq_hz)))
                burst_hz = burst_freq_hz * direction
                logger.debug(
                    "align_yaw burst: error_deg=%.3f cmd_deg=%.3f burst_hz=%d duration=%.3f pulses=%d",
                    error_deg,
                    cmd_deg,
                    burst_hz,
                    burst_duration_s,
                    burst_pulses,
                )

                if not self.pico.set_yaw_velocity(burst_hz, ack_timeout=0.15):
                    logger.warning("align_yaw: failed to start yaw burst")
                    return False

                time.sleep(burst_duration_s)
                self.pico.set_yaw_velocity(0, ack_timeout=0.05)

                # Let the camera thread update before measuring the next chunk.
                time.sleep(YAW_ALIGN_LOOP_S)

            logger.warning("align_yaw: timed out after %.1fs", timeout)
            return False
        finally:
            if locked_here:
                self.detector.unlock_target()

    def set_pitch_for_distance(self, distance_m: float, timeout: float = 8.0) -> bool:
        """
        Look up pitch elevation angle for the given distance, compute delta
        from current position, and send the PITCH command.
        """
        target_pitch, throttle1, throttle2 = interpolate_trajectory(distance_m)
        delta_deg = target_pitch - self._current_pitch_deg

        logger.info(
            "set_pitch: dist=%.2fm → target_pitch=%.1f° (delta=%.1f°) throttle1=%d throttle2=%d",
            distance_m, target_pitch, delta_deg, throttle1, throttle2,
        )

        # Update flywheel speed to match distance if armed
        if self._armed:
            self.pico.set_spin(throttle1, throttle2)

        if not PITCH_ENABLE_MOTION:
            logger.warning(
                "set_pitch: motion bypassed (PITCH_ENABLE_MOTION=False) — using current pitch %.2f°",
                self._current_pitch_deg,
            )
            return True

        # Quantize to whole stepper steps so we never send a zero-step move,
        # which Pico reports as ERR stepper_config_failed.
        delta_steps = int(round(delta_deg / STEPPER_STEP_ANGLE_DEG))
        cmd_delta_deg = delta_steps * STEPPER_STEP_ANGLE_DEG

        if delta_steps == 0:
            logger.debug(
                "Pitch delta %.2f° rounds to 0 steps (step=%.1f°), skipping move",
                delta_deg,
                STEPPER_STEP_ANGLE_DEG,
            )
            return True

        ok = self.pico.move_pitch_sync(cmd_delta_deg, PITCH_FREQ_HZ, timeout)
        if ok:
            self._current_pitch_deg += cmd_delta_deg
            return True

        # Fallback: if Pico is still running old full-step firmware, retry with
        # full-step quantization so command is always representable.
        legacy_steps = int(round(delta_deg / PITCH_LEGACY_STEP_ANGLE_DEG))
        legacy_cmd_deg = legacy_steps * PITCH_LEGACY_STEP_ANGLE_DEG
        if legacy_steps != 0:
            logger.warning(
                "set_pitch: retrying with legacy step angle %.3f° (cmd=%.2f°)",
                PITCH_LEGACY_STEP_ANGLE_DEG,
                legacy_cmd_deg,
            )
            ok_legacy = self.pico.move_pitch_sync(legacy_cmd_deg, PITCH_FALLBACK_FREQ_HZ, timeout)
            if ok_legacy:
                self._current_pitch_deg += legacy_cmd_deg
                return True

        logger.error("Pitch move failed or timed out")
        return False

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

        # Initial detection check + lock this exact cup for the whole shot.
        # Wait briefly for a fresh valid frame so the selected target is stable.
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
            "fire_sequence: selected target x=%.3f y=%.3f conf=%.2f",
            result.x_norm,
            result.y_norm,
            result.confidence,
        )

        try:
            # Align yaw while vision is locked to the selected cup.
            if not self.align_yaw():
                logger.warning("fire_sequence: yaw alignment failed")
                return False

            # Fresh detection after alignment (still locked to same cup).
            # Give the camera a short window to settle so a brief dropout does
            # not cancel an otherwise successful alignment.
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
                logger.warning("fire_sequence: invalid distance estimate %.2fm", distance_m)
                return False

            logger.info("fire_sequence: locked-target distance=%.2fm", distance_m)

            # Set pitch
            if not self.set_pitch_for_distance(distance_m):
                if PITCH_ALLOW_FIRE_WITHOUT_MOVE:
                    logger.warning("fire_sequence: pitch set failed — continuing without pitch move")
                else:
                    logger.warning("fire_sequence: pitch set failed")
                    return False

            # Fire (timed feed)
            logger.info("Firing! feed_on_time=%.3fs", FEED_ON_TIME_S)
            ok = self.pico.feed_time_sync(duration_s=FEED_ON_TIME_S, timeout=5.0)
            if not ok:
                logger.error("Feed timed out or failed")
            return ok
        finally:
            # Always release lock so next shot can pick a new cup.
            self.detector.unlock_target()

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
        ctrl.arm(throttle1=args.throttle1, throttle2=args.throttle2)
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

    # Live camera feed — runs in a daemon thread so input() isn't blocked.
    _display_stop = threading.Event()

    def _display_loop() -> None:
        while not _display_stop.is_set():
            frame = detector.get_annotated_frame()
            if frame is not None:
                cv2.imshow("Cup Detector", frame)
            cv2.waitKey(1)

    display_thread = threading.Thread(
        target=_display_loop, daemon=True, name="display"
    )
    display_thread.start()

    print("Manual control shell. Commands:")
    print("  home          — zero counters")
    print("  arm [t1] [t2] — spin up flywheels (defaults from constants)")
    print("  disarm        — stop flywheels")
    print("  yaw <deg>     — move yaw")
    print("  pitch <deg>   — move pitch")
    print("  feed <ms>     — feed motor timed (milliseconds)")
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
                if len(parts) > 2:
                    t1 = int(parts[1])
                    t2 = int(parts[2])
                elif len(parts) > 1:
                    t1 = int(parts[1])
                    t2 = t1
                else:
                    t1 = FLYWHEEL_THROTTLE_1
                    t2 = FLYWHEEL_THROTTLE_2
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
                    print("Usage: pitch <degrees>")
                else:
                    pico.move_pitch_sync(float(parts[1]))
            elif cmd == "feed":
                if len(parts) < 2:
                    print("Usage: feed <milliseconds>")
                else:
                    duration_s = float(parts[1]) / 1000.0
                    pico.feed_time_sync(duration_s)
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
        _display_stop.set()
        cv2.destroyAllWindows()
        ctrl.estop()
        detector.stop()
        pico.disconnect()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Beer Pong Turret Controller")
    parser.add_argument("--port",     default="/dev/ttyACM0", help="Pico serial port")
    parser.add_argument("--camera",   type=int,   default=0,    help="Camera index")
    parser.add_argument("--focal",    type=float, default=1050.0, help="Camera focal length (px)")
    parser.add_argument("--shots",    type=int,   default=6,    help="Number of shots in auto mode")
    parser.add_argument("--delay",    type=float, default=1.5,  help="Delay between shots (s)")
    parser.add_argument("--throttle1", type=int, default=FLYWHEEL_THROTTLE_1, help="Flywheel 1 DShot throttle")
    parser.add_argument("--throttle2", type=int, default=FLYWHEEL_THROTTLE_2, help="Flywheel 2 DShot throttle")
    parser.add_argument("--manual",   action="store_true", help="Interactive manual control mode")
    args = parser.parse_args()

    if args.manual:
        _run_manual(args)
    else:
        _run_auto(args)


if __name__ == "__main__":
    main()
