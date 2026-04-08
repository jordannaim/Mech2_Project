"""
pico_comms.py — Serial bridge to the Pico turret controller.

Protocol: text lines over UART, newline-terminated.

Pi → Pico:
  YAW <deg> [freq_hz]
  PITCH <deg> [freq_hz]
  SPIN <t1> <t2>          (DShot throttle 0-2047 each)
  FEED [counts]
    FEED_MS <milliseconds>
  HOME
  STATUS
  ESTOP

Pico → Pi:
  OK
  BUSY YAW | BUSY PITCH | BUSY FEED
  DONE YAW
  DONE PITCH
  DONE FEED
  STATUS yaw_moving=0 pitch_moving=0 feed_active=0 spin1=0 spin2=0
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
    feed_active: bool = False
    spin1: int = 0
    spin2: int = 0


class PicoComms:
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200) -> None:
        self._port = port
        self._baud = baud
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._reader_thread: Optional[threading.Thread] = None

        # Synchronization events
        self._ack = threading.Event()          # set on OK
        self._done_yaw = threading.Event()
        self._done_pitch = threading.Event()
        self._done_feed = threading.Event()
        self._boot = threading.Event()

        # Only one command in flight at a time
        self._send_lock = threading.Lock()

        self._status = PicoStatus()
        self._status_lock = threading.Lock()
        self._last_error_line: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, boot_timeout: float = 3.0) -> bool:
        """Open serial port and start background reader. Returns True on success."""
        try:
            self._serial = serial.Serial(
                self._port,
                self._baud,
                timeout=0.1,
                write_timeout=1.0,
            )
        except serial.SerialException as exc:
            logger.error("Failed to open %s: %s", self._port, exc)
            return False

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="pico-reader"
        )
        self._reader_thread.start()

        # Wait for BOOT message (optional — Pico may already be running)
        if self._boot.wait(timeout=boot_timeout):
            logger.info("Pico BOOT received")
        else:
            logger.info("No BOOT message within %.1fs — assuming Pico already running", boot_timeout)

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
            logger.info("Feed complete")
        elif line.startswith("BUSY"):
            logger.warning("Pico busy: %s", line)
            # Still counts as acknowledgment so caller isn't stuck
            self._ack.set()
        elif line.startswith("STATUS"):
            self._parse_status(line)
            # STATUS is also an ack
            self._ack.set()
        elif line.startswith("ERR"):
            self._last_error_line = line
            logger.error("Pico error: %s", line)
            self._ack.set()
        else:
            logger.debug("Unhandled Pico line: %s", line)

    def _parse_status(self, line: str) -> None:
        # STATUS yaw_moving=0 pitch_moving=0 feed_active=0 spin1=0 spin2=0
        fields: dict[str, str] = {}
        for token in line.split()[1:]:
            if "=" in token:
                k, v = token.split("=", 1)
                fields[k] = v
        with self._status_lock:
            self._status.yaw_moving = fields.get("yaw_moving", "0") == "1"
            self._status.pitch_moving = fields.get("pitch_moving", "0") == "1"
            self._status.feed_active = fields.get("feed_active", "0") == "1"
            self._status.spin1 = int(fields.get("spin1", "0"))
            self._status.spin2 = int(fields.get("spin2", "0"))

    # ------------------------------------------------------------------
    # Internal command sender
    # ------------------------------------------------------------------

    def _send(self, cmd: str, ack_timeout: float = 1.0, warn_on_timeout: bool = True) -> bool:
        """Send a command line, wait for OK/ACK. Returns True if acked."""
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

            # If Pico acknowledged with ERR, treat command as failed.
            if self._last_error_line is not None:
                return False
            return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_yaw_velocity(self, freq_hz: int, ack_timeout: float = 0.03) -> bool:
        """
        Run yaw continuously at freq_hz with no step limit.
        Positive = right, negative = left, 0 = stop.
        Non-blocking — no DONE event will be sent by the Pico.
        Used for smooth vision-based alignment.
        """
        # YAW_VEL is often used in a tight loop, so keep the default ACK wait
        # short, but allow callers to request a longer burst-start window.
        # Don't spam warnings for frequent stop commands while aligning.
        warn = (ack_timeout >= 0.05) and (freq_hz != 0)
        return self._send(f"YAW_VEL {freq_hz}", ack_timeout=ack_timeout, warn_on_timeout=warn)

    def move_yaw(self, degrees: float, freq_hz: int = 500) -> bool:
        """Start yaw move. Returns True when Pico acknowledges. Non-blocking."""
        self._done_yaw.clear()
        return self._send(f"YAW {degrees:.2f} {freq_hz}")

    def move_pitch(self, degrees: float, freq_hz: int = 500) -> bool:
        """Start pitch move. Returns True when Pico acknowledges. Non-blocking."""
        self._done_pitch.clear()
        return self._send(f"PITCH {degrees:.2f} {freq_hz}")

    def wait_yaw(self, timeout: float = 10.0) -> bool:
        """Block until DONE YAW or timeout. Returns True on success."""
        result = self._done_yaw.wait(timeout=timeout)
        if not result:
            logger.warning("Timed out waiting for DONE YAW (%.1fs)", timeout)
        return result

    def wait_pitch(self, timeout: float = 10.0) -> bool:
        """Block until DONE PITCH or timeout. Returns True on success."""
        result = self._done_pitch.wait(timeout=timeout)
        if not result:
            logger.warning("Timed out waiting for DONE PITCH (%.1fs)", timeout)
        return result

    def wait_feed(self, timeout: float = 5.0) -> bool:
        """Block until DONE FEED or timeout. Returns True on success."""
        result = self._done_feed.wait(timeout=timeout)
        if not result:
            logger.warning("Timed out waiting for DONE FEED (%.1fs)", timeout)
        return result

    def move_yaw_sync(self, degrees: float, freq_hz: int = 500, timeout: float = 10.0) -> bool:
        """Send YAW and block until move completes."""
        if not self.move_yaw(degrees, freq_hz):
            return False
        return self.wait_yaw(timeout)

    def move_pitch_sync(self, degrees: float, freq_hz: int = 500, timeout: float = 10.0) -> bool:
        """Send PITCH and block until move completes."""
        if not self.move_pitch(degrees, freq_hz):
            return False
        return self.wait_pitch(timeout)

    def set_spin(self, throttle1: int, throttle2: int) -> bool:
        """Set flywheel DShot throttles (0-2047 each)."""
        t1 = max(0, min(2047, throttle1))
        t2 = max(0, min(2047, throttle2))
        return self._send(f"SPIN {t1} {t2}")

    def feed(self, counts: Optional[int] = None) -> bool:
        """Trigger feed motor. Non-blocking after Pico ACK."""
        self._done_feed.clear()
        cmd = f"FEED {counts}" if counts is not None else "FEED"
        return self._send(cmd)

    def feed_sync(self, counts: Optional[int] = None, timeout: float = 5.0) -> bool:
        """Feed and block until DONE FEED."""
        if not self.feed(counts):
            return False
        return self.wait_feed(timeout)

    def feed_time(self, duration_s: float) -> bool:
        """
        Trigger feed for a fixed on-time in seconds. Non-blocking after Pico ACK.
        Requires Pico firmware support for FEED_MS.
        """
        self._done_feed.clear()
        ms = max(1, int(round(duration_s * 1000.0)))
        if self._send(f"FEED_MS {ms}"):
            return True

        # Backward compatibility: older firmware may not implement FEED_MS.
        if self._last_error_line == "ERR unknown_command":
            # Approximate conversion based on historical default: 200 counts
            # over ~0.75 s => ~267 counts/s.
            fallback_counts = max(1, int(round(duration_s * 267.0)))
            logger.warning(
                "FEED_MS unsupported by firmware; falling back to FEED %d counts",
                fallback_counts,
            )
            return self.feed(fallback_counts)

        return False

    def feed_time_sync(self, duration_s: float, timeout: float = 5.0) -> bool:
        """Timed feed and block until DONE FEED."""
        if not self.feed_time(duration_s):
            return False
        return self.wait_feed(timeout)

    def home(self) -> bool:
        """Zero position counters (no motion)."""
        return self._send("HOME")

    def estop(self) -> bool:
        """Emergency stop all motors."""
        # ESTOP is urgent — bypass the send lock if needed
        if not self._serial or not self._serial.is_open:
            return False
        try:
            self._serial.write(b"ESTOP\n")
            self._serial.flush()
        except serial.SerialException:
            return False
        return True

    def get_status(self, timeout: float = 1.0) -> PicoStatus:
        """Query Pico status. Returns latest known status."""
        self._send("STATUS", ack_timeout=timeout)
        with self._status_lock:
            return PicoStatus(
                yaw_moving=self._status.yaw_moving,
                pitch_moving=self._status.pitch_moving,
                feed_active=self._status.feed_active,
                spin1=self._status.spin1,
                spin2=self._status.spin2,
            )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PicoComms":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.estop()
        self.disconnect()
