"""
pico_comms.py — Serial bridge to the Pico turret controller.

Protocol: text lines over USB CDC, newline-terminated.

Pi → Pico:
  YAW_VEL <signed_freq_hz>
  YAW <deg> [freq_hz]
  PITCH <deg> [freq_hz]
  PITCH_STEPS <n> [freq_hz]
  FIRE <steps> [freq_hz]
  SPIN <t1> <t2>          (DShot throttle 0-2047 each)
  HOME
  STATUS
  ESTOP

Pico → Pi:
  OK
  BUSY YAW | BUSY PITCH | BUSY FEED
  DONE YAW
  DONE PITCH
  DONE FEED
  STATUS yaw_moving=0 pitch_moving=0 indexer_moving=0 fire_active=0 spin1=0 spin2=0
  ERR <reason>
  BOOT
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial

logger = logging.getLogger(__name__)


@dataclass
class PicoStatus:
    yaw_moving: bool = False
    pitch_moving: bool = False
    indexer_moving: bool = False
    fire_active: bool = False
    feed_active: bool = False   # alias for fire_active (backward compat)
    spin1: int = 0
    spin2: int = 0


class PicoComms:
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200) -> None:
        self._port = port
        self._baud = baud
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

        self._ack        = threading.Event()
        self._done_yaw   = threading.Event()
        self._done_pitch = threading.Event()
        self._done_feed  = threading.Event()
        self._boot       = threading.Event()

        self._send_lock = threading.Lock()

        self._status = PicoStatus()
        self._status_lock = threading.Lock()
        self._last_error_line: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, boot_timeout: float = 3.0) -> bool:
        try:
            self._serial = serial.Serial(
                self._port, self._baud, timeout=0.1, write_timeout=1.0,
            )
        except serial.SerialException as exc:
            logger.error("Failed to open %s: %s", self._port, exc)
            return False

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="pico-reader"
        )
        self._reader_thread.start()

        if self._boot.wait(timeout=boot_timeout):
            logger.info("Pico BOOT received")
        else:
            logger.info("No BOOT within %.1fs — assuming Pico already running", boot_timeout)

        return True

    def disconnect(self) -> None:
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        logger.info("Disconnected from Pico")

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        buf = b""
        while self._running:
            try:
                chunk = self._serial.read(64)
            except serial.SerialException as exc:
                logger.error("Serial read error: %s", exc)
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode("ascii", errors="replace").strip()
                if line:
                    self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        logger.debug("Pico → Pi: %s", line)

        if line == "OK":
            self._ack.set()
        elif line == "BOOT":
            self._boot.set()
            logger.info("Pico booted")
        elif line == "DONE YAW":
            self._done_yaw.set()
            logger.info("Yaw move complete")
        elif line == "DONE PITCH":
            self._done_pitch.set()
            logger.info("Pitch move complete")
        elif line == "DONE FEED":
            self._done_feed.set()
            logger.info("Fire/feed complete")
        elif line.startswith("BUSY"):
            logger.warning("Pico busy: %s", line)
            self._ack.set()
        elif line.startswith("STATUS"):
            self._parse_status(line)
            self._ack.set()
        elif line.startswith("ERR"):
            self._last_error_line = line
            logger.error("Pico error: %s", line)
            self._ack.set()
        else:
            logger.debug("Unhandled Pico line: %s", line)

    def _parse_status(self, line: str) -> None:
        fields: dict[str, str] = {}
        for token in line.split()[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                fields[k] = v
        with self._status_lock:
            self._status.yaw_moving     = fields.get("yaw_moving", "0")     == "1"
            self._status.pitch_moving   = fields.get("pitch_moving", "0")   == "1"
            self._status.indexer_moving = fields.get("indexer_moving", "0") == "1"
            fire_active = fields.get("fire_active", fields.get("feed_active", "0")) == "1"
            self._status.fire_active  = fire_active
            self._status.feed_active  = fire_active  # alias
            self._status.spin1 = int(fields.get("spin1", "0"))
            self._status.spin2 = int(fields.get("spin2", "0"))

    # ------------------------------------------------------------------
    # Internal sender
    # ------------------------------------------------------------------

    def _send(self, cmd: str, ack_timeout: float = 1.0, warn_on_timeout: bool = True) -> bool:
        if not self._serial or not self._serial.is_open:
            logger.error("Serial not open")
            return False

        with self._send_lock:
            self._ack.clear()
            self._last_error_line = None
            raw = (cmd.strip() + "\n").encode("ascii")
            try:
                self._serial.write(raw)
                self._serial.flush()
            except serial.SerialException as exc:
                logger.error("Write error: %s", exc)
                return False

            if not self._ack.wait(timeout=ack_timeout):
                if warn_on_timeout:
                    logger.warning("No ACK for command: %s", cmd)
                return False

            if self._last_error_line is not None:
                return False
            return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_yaw_velocity(self, freq_hz: int, ack_timeout: float = 0.03) -> bool:
        warn = (ack_timeout >= 0.05) and (freq_hz != 0)
        return self._send(f"YAW_VEL {freq_hz}", ack_timeout=ack_timeout, warn_on_timeout=warn)

    def move_yaw(self, degrees: float, freq_hz: int = 500) -> bool:
        self._done_yaw.clear()
        return self._send(f"YAW {degrees:.2f} {freq_hz}")

    def move_pitch(self, degrees: float, freq_hz: int = 500) -> bool:
        self._done_pitch.clear()
        return self._send(f"PITCH {degrees:.2f} {freq_hz}")

    def move_pitch_steps(self, steps: int, freq_hz: int = 400) -> bool:
        """Send PITCH_STEPS <steps> — non-blocking after ACK."""
        self._done_pitch.clear()
        return self._send(f"PITCH_STEPS {steps} {freq_hz}")

    def wait_yaw(self, timeout: float = 10.0) -> bool:
        result = self._done_yaw.wait(timeout=timeout)
        if not result:
            logger.warning("Timed out waiting for DONE YAW (%.1fs)", timeout)
        return result

    def wait_pitch(self, timeout: float = 10.0) -> bool:
        result = self._done_pitch.wait(timeout=timeout)
        if not result:
            logger.warning("Timed out waiting for DONE PITCH (%.1fs)", timeout)
        return result

    def wait_feed(self, timeout: float = 10.0) -> bool:
        result = self._done_feed.wait(timeout=timeout)
        if not result:
            logger.warning("Timed out waiting for DONE FEED (%.1fs)", timeout)
        return result

    def move_yaw_sync(self, degrees: float, freq_hz: int = 500, timeout: float = 10.0) -> bool:
        if not self.move_yaw(degrees, freq_hz):
            return False
        return self.wait_yaw(timeout)

    def move_pitch_sync(self, degrees: float, freq_hz: int = 500, timeout: float = 15.0) -> bool:
        if not self.move_pitch(degrees, freq_hz):
            return False
        return self.wait_pitch(timeout)

    def move_pitch_steps_sync(self, steps: int, freq_hz: int = 400, timeout: float = 15.0) -> bool:
        """Send PITCH_STEPS and block until DONE PITCH."""
        if not self.move_pitch_steps(steps, freq_hz):
            return False
        return self.wait_pitch(timeout)

    def set_spin(self, throttle1: int, throttle2: int) -> bool:
        t1 = max(0, min(2047, throttle1))
        t2 = max(0, min(2047, throttle2))
        return self._send(f"SPIN {t1} {t2}")

    def fire(self, steps: int) -> bool:
        """Send FIRE <steps> — non-blocking after ACK."""
        self._done_feed.clear()
        return self._send(f"FIRE {steps}")

    def fire_sync(self, steps: int, timeout: float = 8.0) -> bool:
        """Send FIRE <steps> and block until DONE FEED (full indexer + servo sequence)."""
        if not self.fire(steps):
            return False
        return self.wait_feed(timeout)

    def feed(self, counts: Optional[int] = None) -> bool:
        """Trigger fire sequence (backward-compat; uses FIRE command)."""
        self._done_feed.clear()
        return self._send("FEED")

    def feed_sync(self, counts: Optional[int] = None, timeout: float = 8.0) -> bool:
        if not self.feed(counts):
            return False
        return self.wait_feed(timeout)

    def feed_time(self, duration_s: float) -> bool:
        """Trigger fire sequence (backward-compat; uses FIRE command)."""
        self._done_feed.clear()
        return self._send("FEED_MS 420")

    def feed_time_sync(self, duration_s: float, timeout: float = 8.0) -> bool:
        """Backward-compat: maps to fire_sync with default step count."""
        if not self.feed_time(duration_s):
            return False
        return self.wait_feed(timeout)

    def home(self) -> bool:
        return self._send("HOME")

    def estop(self) -> bool:
        if not self._serial or not self._serial.is_open:
            return False
        try:
            self._serial.write(b"ESTOP\n")
            self._serial.flush()
        except serial.SerialException:
            return False
        return True

    def get_status(self, timeout: float = 1.0) -> PicoStatus:
        self._send("STATUS", ack_timeout=timeout)
        with self._status_lock:
            return PicoStatus(
                yaw_moving=self._status.yaw_moving,
                pitch_moving=self._status.pitch_moving,
                indexer_moving=self._status.indexer_moving,
                fire_active=self._status.fire_active,
                feed_active=self._status.fire_active,
                spin1=self._status.spin1,
                spin2=self._status.spin2,
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PicoComms":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.estop()
        self.disconnect()
