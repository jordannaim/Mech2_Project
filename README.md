# Beer Pong Turret — System Documentation

Fully autonomous beer pong shooting robot. A Raspberry Pi runs computer vision to detect cups and compute aim, a Raspberry Pi Pico drives the stepper motors and flywheel ESCs, and the two talk to each other over a USB serial cable.

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Software Requirements](#2-software-requirements)
3. [Repository Structure](#3-repository-structure)
4. [Flashing the Pico](#4-flashing-the-pico)
5. [Setting Up the Pi](#5-setting-up-the-pi)
6. [Running the System](#6-running-the-system)
7. [How the Pi Thinks — Step by Step](#7-how-the-pi-thinks--step-by-step)
8. [Calibration](#8-calibration)
9. [What Happens When No Cups Are Detected](#9-what-happens-when-no-cups-are-detected)
10. [Tuning Reference](#10-tuning-reference)

---

## 1. Hardware Requirements

| Component | Purpose |
|---|---|
| Raspberry Pi (3B+ or 4) | Vision processing and high-level control |
| Raspberry Pi Pico | Real-time motor control |
| USB micro-B cable | Pi ↔ Pico communication and Pico power |
| USB webcam | Cup detection (mounted on yaw axis) |
| Stepper motor × 2 | Yaw (pan) and pitch (tilt) |
| Stepper driver × 2 | Current control for steppers |
| Brushless drone motor × 2 | Flywheel launcher |
| ESC × 2 (DShot capable) | Flywheel motor control |
| DC motor + encoder | Ball feed mechanism |
| 16V power rail | Drone motors and ESCs |
| 5V power | Pi and Pico logic |

### Wiring (Pico pin assignments)

| Signal | GPIO |
|---|---|
| Stepper YAW — STEP | GPIO 27 |
| Stepper YAW — DIR | GPIO 26 |
| Stepper PITCH — STEP | GPIO 17 |
| Stepper PITCH — DIR | GPIO 16 |
| DShot Flywheel Motor 1 | GPIO 12 |
| DShot Flywheel Motor 2 | GPIO 13 |
| Feed Motor PWM | GPIO 14 |
| Feed Motor DIR | GPIO 15 |
| Feed Encoder A | GPIO 18 |
| Pi communication | USB (micro-B connector) |

> **Note:** There are no UART wires between the Pi and Pico. All communication goes through the single USB cable. The Pico appears as `/dev/ttyACM0` on the Pi.

---

## 2. Software Requirements

### Pico (build machine — Windows or Linux with ARM toolchain)

- [Raspberry Pi Pico SDK](https://github.com/raspberrypi/pico-sdk)
- CMake ≥ 3.13
- ARM GCC toolchain (`arm-none-eabi-gcc`)
- `PICO_SDK_PATH` environment variable set to your SDK location

### Raspberry Pi

```bash
sudo apt update
sudo apt install python3-pip python3-opencv
pip3 install pyserial
```

Python ≥ 3.10 is required.

---

## 3. Repository Structure

```
Code/
├── pico/
│   ├── main.c          — Pico firmware (motors, DShot, USB serial protocol)
│   └── CMakeLists.txt  — Build configuration
├── pi/
│   ├── pico_comms.py   — Serial bridge to Pico (blocking waits for motor done)
│   ├── vision.py       — Cup detection pipeline (OpenCV)
│   └── controller.py   — Main orchestration: detect → align → aim → fire
└── README.md
```

The `dshot.pio` file is referenced from the separate DSHOT project. Either copy it into `pico/` and update the CMakeLists.txt path, or leave the path pointing to its current location.

---

## 4. Flashing the Pico

### Step 1 — Copy dshot.pio

Copy `dshot.pio` from `C:\Users\davee\OneDrive\Desktop\DSHOT\` into the `pico/` folder so the build is self-contained. Then open `pico/CMakeLists.txt` and change the `pico_generate_pio_header` line to:

```cmake
pico_generate_pio_header(turret ${CMAKE_CURRENT_LIST_DIR}/dshot.pio)
```

### Step 2 — Build

```bash
cd "Semester 8/Mechatronics2/Code/pico"
mkdir build && cd build
cmake .. -DPICO_SDK_PATH=$PICO_SDK_PATH
make -j4
```

This produces `turret.uf2` in the `build/` folder.

### Step 3 — Flash

1. Hold the **BOOTSEL** button on the Pico
2. Plug it into your computer via USB while holding BOOTSEL
3. Release BOOTSEL — the Pico mounts as a USB drive called `RPI-RP2`
4. Drag and drop `turret.uf2` onto the `RPI-RP2` drive
5. The Pico reboots automatically and starts running the firmware

### Verify the flash

Open any serial terminal (e.g. PuTTY, VS Code serial monitor, or `python -m serial.tools.miniterm /dev/ttyACM0 115200`) and connect to the Pico's USB port. You should see:

```
BOOT
```

printed once as soon as the Pi-side serial port is opened.

---

## 5. Setting Up the Pi

### Transfer the Pi code

Copy the `pi/` folder to the Raspberry Pi. Any method works — USB drive, `scp`, shared network folder.

### Verify the camera

```bash
ls /dev/video*
```

The USB camera should appear as `/dev/video0`. Test it with:

```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Verify the Pico serial port

Plug in the Pico over USB and run:

```bash
ls /dev/ttyACM*
```

It should show `/dev/ttyACM0`. If it appears as `ttyACM1`, pass `--port /dev/ttyACM1` when running the controller.

### Permissions (one-time setup)

```bash
sudo usermod -a -G dialout $USER
```

Log out and back in for this to take effect. This allows your user to access the serial port without `sudo`.

---

## 6. Running the System

### Physical setup before every run

1. **Pitch home position**: Manually tilt the launcher all the way down until it physically rests against the base stop. This is pitch = 0°. The software tracks all pitch motion as a delta from this position.
2. **Point the turret roughly toward the table** — within the camera's field of view of the cups.
3. Power on the motor power rail (16V for ESCs/flywheels).
4. Plug the Pico into the Pi via USB.

### Auto mode (fully autonomous)

```bash
cd /path/to/Code
python3 -m pi.controller
```

Optional arguments:
```
--port /dev/ttyACM0    Pico serial port (default: /dev/ttyACM0)
--camera 0             Camera index (default: 0)
--focal 700.0          Camera focal length in pixels (calibrate this — see §8)
--shots 6              Number of shots to attempt (default: 6)
--delay 1.5            Seconds between shots (default: 1.5)
--throttle 800         Flywheel DShot throttle 0-2047 (default: 800)
```

### Manual / debug mode

```bash
python3 -m pi.controller --manual
```

This opens an interactive shell:

```
> home          — zero Pico position counters
> arm           — spin up flywheels
> arm 900       — spin up at throttle 900
> disarm        — stop flywheels
> yaw 15.0      — move yaw 15 degrees right
> yaw -10.0     — move yaw 10 degrees left
> pitch 8.0     — elevate pitch 8 degrees
> align         — run yaw auto-alignment to center the detected cup
> fire          — run full fire sequence (align + pitch + feed)
> status        — query Pico motor state
> detect        — print latest vision detection result
> estop         — emergency stop all motors
> quit
```

Manual mode is the right starting point for calibration: use `detect` to confirm vision is working, `yaw`/`pitch` to verify motor directions, and `fire` to test the complete sequence once everything looks right.

---

## 7. How the Pi Thinks — Step by Step

This section explains the complete decision-making chain from boot to shot.

### 7.1 Startup

When you run the controller, three things initialize in parallel:

- **PicoComms** opens `/dev/ttyACM0` and starts a background reader thread. That thread continuously reads lines from the Pico and parses them. When it sees `DONE YAW`, `DONE PITCH`, or `DONE FEED`, it sets a `threading.Event` — this is how the Pi knows a motor move has finished without polling.
- **CupDetector** opens the camera and starts a background capture thread. That thread continuously grabs frames and runs the full detection pipeline, storing the latest result. The main thread can call `get_result()` at any time and get the most recently processed frame's data.
- **TurretController** is created, linking the two above.

Then `home()` is called — this sends `HOME` to the Pico, which zeroes its step counters. No motion happens. It just tells both sides "this is position zero."

Then `arm()` is called — sends `SPIN 800 800` to the Pico (or whatever `--throttle` was set to). The Pico starts sending that DShot value to both ESCs every millisecond. The Pi then waits `FLYWHEEL_SPINUP_S` (2 seconds) for the motors to reach speed before attempting any shots.

---

### 7.2 The Vision Pipeline

The camera background thread runs this sequence on every frame:

**Step 1 — Red ROI detection**

The frame is converted from BGR to HSV color space. HSV is used instead of RGB because hue (the H channel) is mostly independent of lighting brightness, making red cup detection much more consistent as lighting changes.

Red wraps around the HSV hue wheel (it exists near hue 0 AND near hue 180), so two masks are created and combined:
- Hue 0–8, saturation > 150, value > 80
- Hue 172–180, saturation > 150, value > 80

The combined mask is cleaned up with morphological operations (open to remove noise, close to fill gaps, dilate to merge nearby blobs). The bounding rectangle of all red blobs above a minimum area is computed and expanded by 60 pixels of padding. Everything outside this bounding box is ignored for the rest of the frame.

**Why do this first?** Ellipse detection on the full 1280×720 frame is expensive and would find many false positives. Narrowing down to a red region first makes the detection both faster and more accurate.

**Step 2 — Edge detection**

Inside the red ROI, the image is converted to grayscale, blurred twice with a 5×5 Gaussian (double blur reduces noise further while preserving real edges), and Canny edge detection is run with low threshold 60 and high threshold 120. The resulting edge map is morphologically closed (connects nearby edge fragments) and dilated slightly (thickens edges to improve the ellipse support scoring later).

**Step 3 — Ellipse fitting**

`findContours` finds all connected edge curves. For each contour with at least 12 points, `fitEllipse` fits the best-fit ellipse to those points. Each candidate ellipse is then filtered:

- **Major axis** must be 40–240 pixels — filters out tiny noise and objects larger than a cup
- **Minor axis** must be 12–180 pixels
- **Aspect ratio** (minor/major) must be 0.18–1.0 — a cup rim viewed from an angle appears as a narrow ellipse; ratio below 0.18 means it's seen nearly edge-on and can't be reliably targeted

**Step 4 — Scoring**

For each candidate that passes the geometric filters, a **support score** is computed: 120 points are sampled evenly around the ellipse perimeter, and the fraction of those points that land within 2 pixels of a real edge in the edge map is measured. A high support score means the ellipse boundary actually lines up with real edges in the image — not just a mathematically coincidental fit to a random contour.

The final score blends support with how close the aspect ratio is to 0.6 (a typical cup rim viewed at a moderate angle):

```
score = 0.7 × support_fraction + 0.3 × (1 - |aspect - 0.6|)
```

Candidates with score below 0.30 are discarded. The highest-scoring candidate is chosen as the target.

**Step 5 — Output**

From the best candidate:
- `x_norm = (cup_center_x - frame_width/2) / (frame_width/2)` — ranges from -1.0 (left edge) to +1.0 (right edge), 0.0 = horizontally centered
- `y_norm = (frame_height/2 - cup_center_y) / (frame_height/2)` — ranges -1.0 (bottom) to +1.0 (top), note Y is flipped because image coordinates go down
- `distance_m` — estimated using the pinhole camera model: `distance = (real_diameter_m × focal_length_px) / apparent_major_axis_px`. Since the cup rim diameter is a known constant (90mm), and the focal length is calibrated once, the apparent size of the ellipse directly gives distance.
- `confidence` — the final score (0.0–1.0)

This result is stored in a thread-safe variable and updated on every frame.

---

### 7.3 The Fire Sequence

When `fire_sequence()` is called (once per shot in auto mode):

**Check 1 — Is there a target?**

The latest detection result is fetched. If `valid = False` or `confidence < 0.35`, the shot is aborted immediately. The system is not armed but it doesn't crash — it logs a warning and `run_auto` tries again after the delay.

**Yaw alignment loop**

This is the most important step and runs iteratively:

1. Read `x_norm` from the latest detection
2. If `|x_norm| < 0.04` (the dead zone, roughly 2.4° given 60° FOV), the cup is centered — stop
3. Otherwise, compute how many degrees to move: `delta_deg = x_norm × (60° / 2) = x_norm × 30°`
   - Example: cup is 20% to the right → `x_norm = 0.2` → move yaw 6° right
4. Send `YAW 6.0 300` to the Pico. The Pico responds `OK` immediately, then sends `DONE YAW` when the move is physically complete. The Pi blocks on `DONE YAW` before proceeding.
5. Wait 0.15 seconds for the camera to capture a fresh frame (the camera is mounted on the yaw axis, so the view shifts with each move)
6. Repeat up to 10 iterations

**Why iterative?** Because stepper motors are open loop — there's no guarantee the actual mechanical movement matches the commanded degrees exactly (backlash, load variation). Each correction is based on what the camera actually sees after the previous move, not what was commanded. This closes the loop using vision.

**Re-detect for distance**

After yaw is aligned, a fresh detection result is fetched to get an accurate distance estimate. The yaw alignment may have taken several moves and the distance reading from before alignment is stale.

**Pitch and throttle calculation**

`interpolate_trajectory(distance_m)` looks up the `TRAJECTORY_TABLE`. This table maps measured distance to two outputs: the absolute pitch elevation angle (degrees up from home) and the DShot throttle value for the flywheels.

Linear interpolation is used between the two nearest table entries. If the distance is outside the table range, the nearest endpoint is used (clamped).

The pitch *delta* is computed as `target_pitch_abs - current_pitch_deg`. The controller tracks absolute pitch position as a running sum of all commanded moves since homing. If the target is 12° and the turret is currently at 8° (from a previous shot), it sends `PITCH 4.0`. The Pico blocks until the move completes and sends `DONE PITCH`.

The flywheel throttle is updated at the same time via `SPIN <t> <t>`.

**Fire**

`FEED` is sent to the Pico. The Pico pulses the feed motor until its encoder counts 200 pulses (one ball), then stops and sends `DONE FEED`. The Pi waits for `DONE FEED` before marking the shot complete.

---

### 7.4 Between Shots

After each shot, `run_auto` waits `delay_between_shots` (default 1.5 seconds) before starting the next cycle. This gives the ball time to clear the flywheel zone and the turret to stabilize before the next detection is taken.

Pitch position is **not** reset between shots — if the last shot was aimed at 12° and the next cup is also at roughly the same distance, no pitch move happens (delta = 0). The system only moves what needs to move.

---

## 8. Calibration

### Focal length calibration

The distance estimate depends entirely on `focal_length_px`. To calibrate it:

1. Place a single cup at an exactly known distance from the camera (e.g. 1.5 m)
2. Run `python3 -m pi.controller --manual`
3. Type `detect` and note the `distance_m` value that comes out
4. The correct focal length is: `focal_correct = focal_current × (actual_distance / reported_distance)`
5. Pass the corrected value as `--focal <value>` or update the default in `controller.py`

### Trajectory table calibration

The `TRAJECTORY_TABLE` in `controller.py` is the most important tuning step. The default values are rough estimates — they will not be correct for your specific flywheel gap, motor speed, and ball weight variation.

**Method:**
1. Run in manual mode
2. `home`, `arm`, then `pitch <deg>` and `feed` at each distance
3. Record the (distance, pitch, throttle) combinations that consistently land in a cup
4. Replace the table entries in `controller.py`

Work from close to far — shorter distances first since they're easier to verify and give you a baseline for the model.

### Encoder counts per ball

`FEED_DEFAULT_COUNTS = 200` in `pico/main.c` sets how many encoder pulses constitute one ball feed. Adjust this so the mechanism delivers exactly one ball per `FEED` command. You can override it per-call with `FEED <n>` in the serial protocol or `pico.feed(counts=N)` in Python.

---

## 9. What Happens When No Cups Are Detected

Currently, when `vision.py` finds no red regions or no valid ellipses, it sets `DetectionResult.valid = False`. The controller responds as follows:

**During yaw alignment** — `align_yaw()` reads the detection, sees `valid = False`, logs a warning, and returns `False`. The fire sequence is aborted for that shot.

**At the fire sequence entry check** — `fire_sequence()` checks validity before doing anything. If no target, it returns `False` immediately.

**In the auto loop** — `run_auto()` calls `run_once()`, gets `False`, and simply waits `delay_between_shots` before trying again. The system keeps attempting shots until it either detects something or exhausts the `shots` count.

**There is no active search behavior.** If the turret is pointed away from the cups entirely, it will sit and retry indefinitely without moving. This is a gap in the current implementation.

A practical search behavior to add would be: if `align_yaw()` returns False due to no target after N consecutive attempts, sweep the yaw motor in small increments (e.g. 5° steps across the expected table arc) until a detection appears, then proceed. This is straightforward to add to `align_yaw()` using the existing `move_yaw_sync()` calls.

For a demo setting where the turret is always pre-pointed at the table, the current behavior is usually fine — the cups are nearly always in frame and the system recovers on the next cycle if a detection is briefly lost (e.g. someone's hand passes in front of the camera).

---

## 10. Tuning Reference

| Parameter             | File                | Default   | Effect                                                         |
| --------------------- | ------------------- | --------- | -------------------------------------------------------------- |
| `CAMERA_HFOV_DEG`     | controller.py       | 60.0      | Maps x_norm to yaw degrees. Measure your lens's actual FOV.    |
| `YAW_DEAD_ZONE`       | controller.py       | 0.04      | Fraction of half-frame. Tighter = more precise but more moves. |
| `MIN_CONFIDENCE`      | controller.py       | 0.35      | Raise if getting false positives, lower if missing cups.       |
| `FLYWHEEL_THROTTLE`   | controller.py       | 800       | Base throttle. Overridden per-distance by TRAJECTORY_TABLE.    |
| `FLYWHEEL_SPINUP_S`   | controller.py       | 2.0       | Increase if balls are launching sluggishly at the start.       |
| `YAW_ALIGN_FREQ_HZ`   | controller.py       | 300       | Lower = more precise yaw moves, slower alignment.              |
| `FEED_DEFAULT_COUNTS` | pico/main.c         | 200       | Encoder counts per ball. Adjust for your feed mechanism.       |
| `TRAJECTORY_TABLE`    | controller.py       | estimates | **Must be calibrated empirically.** See §8.                    |
| `focal_length_px`     | vision.py / --focal | 700.0     | **Must be calibrated.** See §8.                                |
