
At the moment the code works decently well, but some major changes need to be made. We made a PCB for our electrical system which changed some of the pinouts. Below is the pinout of the pico and what it controls:

**GP0 - Drone motor 1 (DSHOT)**
**GP1 - Drone motor 2 (DSHOT)**

**Pitch Stepper:**
**GP20 - Direction**
**GP21 - STEP**

**Yaw Stepper:**
**GP26 - Direction**
**GP27 - STEP**

**Indexer Stepper (NEW):**
**GP16 - Direction**
**GP17 - STEP**

**Servo:**
**GP4 - 500-2500us Pulse Width Control Signal**



Stepper Motors and Controllers:
Motor Drivers:
https://biqu.equipment/products/bigtreetech-tmc2209-stepper-motor-driver-for-3d-printer-board-vs-tmc2208?srsltid=AfmBOoq1CNPl4jsik7IA77OqxvKsVOyxWU2wH3O0Db6ERgvXJ3zxsUuI

At the moment we have both the MS1 and MS2 pins pulled high which should give a resolution of 16 microsteps. 

Yaw Stepper Motor:
https://www.omc-stepperonline.com/nema-23-stepper-motor-bipolar-1-8deg-3-00nm-424-83oz-in-5-0a-57x57x100mm-4-wires-23hs39-5004s?search=23HS39-5004s

Pitch Stepper Motor + Gearbox:
https://www.omc-stepperonline.com/nema-17-bipolar-1-8deg-65ncm-92oz-in-2-1a-3-36v-42x42x60mm-4-wires-17hs24-2104s
https://www.omc-stepperonline.com/eg-series-planetary-gearbox-gear-ratio-10-1-backlash-15-arc-min-for-nema-17-stepper-motor-eg17-g10?search=EG17-G10

Indexer Stepper:
https://www.omc-stepperonline.com/nema-17-bipolar-0-9deg-11ncm-15-6oz-in-1-2a-3-6v-42x42x21mm-4-wires-17hm08-1204s

We are using these exact motors and drivers be sure to really research how to use them together so that they work seamlessly with the code.

**Overall System Architecture:**
The system uses the camera CV to find a cup which it deems as the target, then it estimates how far it is. Using this estimation it uses a calibration to figure out how far to pitch and how fast to spin each motor. Then it initializes a fire command, **This is what is going to change the most**: We now have a combination of things to do on the fire command. We have a stepper motor which indexes a ball into position which is then fed into the flywheels with a servo. Basic command: Stepper moves a certain number of steps (will need to be calibrated) then the servo is sent a 2500us pulse width command to feed the ball in then the servo retracts back to 500us pulse width command. The 500us position is its home position as the indexer stepper motor hits the servo if it is not in the 500us position.

Home state/where it starts: Since the pitch is open loop (so is yaw but the camera closes the loop as we use a type of PID to align our shooter to the cups) we start with it fully leaned back. The system then needs to apply a positive amount of steps to pitch up. It must keep track of amount of steps both positive (pitch up) and negative (pitch down) to know roughly where it is. Don't even record as an angle just keep track of amount of steps. This will be a big number most likely as it is geared (10k steps is about the increment we move it now from the home position).

What we need to calibrate:
1. Amount of steps the feeder stepper needs to move
2. Speed of drone motors and pitch angle of stepper vs distance of ball launched.
3. Vision system (This works well at the moment just the json needs to get retuned once in a while if the lighting changes drastically)