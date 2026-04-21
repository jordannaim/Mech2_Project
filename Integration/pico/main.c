/**
 * Beer Pong Turret — Pico Firmware
 *
 * Pin assignments (PCB rev 2):
 *   USB CDC (Pi comms): USB micro-B on Pico → USB-A on Pi (/dev/ttyACM0)
 *   DShot Motor 1  : GPIO0
 *   DShot Motor 2  : GPIO1
 *   Servo          : GPIO4   (500-2500 µs, 50 Hz — hardware alarm driven)
 *   Stepper PITCH  : STEP=GPIO21, DIR=GPIO20
 *   Stepper YAW    : STEP=GPIO27, DIR=GPIO26
 *   Stepper INDEXER: STEP=GPIO17, DIR=GPIO16
 *
 * Protocol: text lines, newline-terminated.
 *
 * Pi → Pico commands:
 *   YAW_VEL <signed_freq_hz>      continuous yaw velocity (no step limit)
 *   YAW <deg> [freq_hz]           step-counted yaw move
 *   PITCH <deg> [freq_hz]         step-counted pitch move (output-shaft degrees)
 *   PITCH_STEPS <n> [freq_hz]     pitch move by raw signed step count
 *   FIRE <steps> [freq_hz]        index ball + servo extend/retract
 *   SPIN <t1> <t2>                set flywheel DShot throttle (0-2047 each)
 *   HOME                          zero position counters, no motion
 *   STATUS                        query state
 *   ESTOP                         stop everything, servo to home
 *
 * Pico → Pi responses/events:
 *   OK
 *   BUSY YAW | BUSY PITCH | BUSY FEED
 *   DONE YAW
 *   DONE PITCH
 *   DONE FEED                     sent when full fire sequence completes
 *   STATUS yaw_moving=0 pitch_moving=0 indexer_moving=0 fire_active=0 spin1=0 spin2=0
 *   ERR <reason>
 *   BOOT
 */

#include "pico/stdlib.h"
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "hardware/irq.h"
#include "hardware/timer.h"
#include "dshot.pio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* =========================================================================
 * Pin definitions
 * ========================================================================= */

#define DSHOT_PIN_1         0u
#define DSHOT_PIN_2         1u

#define SERVO_PIN           4u

#define PITCH_STEP_PIN      21u
#define PITCH_DIR_PIN       20u

#define YAW_STEP_PIN        27u
#define YAW_DIR_PIN         26u

#define INDEXER_STEP_PIN    17u
#define INDEXER_DIR_PIN     16u

/* =========================================================================
 * PWM slices  (slice = (gpio/2) % 8)
 *   YAW_STEP  GPIO27: (27/2)%8 = 5
 *   PITCH_STEP GPIO21: (21/2)%8 = 2
 *   INDEXER_STEP GPIO17: (17/2)%8 = 0
 *
 * NOTE: GPIO4 (servo) also maps to slice 2, channel A.
 *       Servo uses hardware alarm instead of PWM to avoid conflict.
 * ========================================================================= */

#define YAW_SLICE       5u
#define PITCH_SLICE     2u
#define INDEXER_SLICE   0u

/* =========================================================================
 * Tuning constants
 * ========================================================================= */

/* Step angles per microstep (MS1=MS2=HIGH → 16 microsteps) */
#define YAW_STEP_ANGLE_DEG      (1.8f / 16.0f)          /* 0.1125° per yaw microstep */
#define PITCH_STEP_ANGLE_DEG    (1.8f / 16.0f / 10.0f)  /* 0.01125° per pitch microstep (output shaft, 10:1 gearbox) */
#define INDEXER_STEP_ANGLE_DEG  (0.9f / 16.0f)          /* 0.05625° per indexer microstep */

#define STEPPER_DEFAULT_FREQ_HZ  500u
#define YAW_VEL_RAMP_HZ_PER_MS  30

#define DSHOT_BITRATE_HZ         150000u
#define DSHOT_THROTTLE_MAX       2047u
#define DSHOT_SEND_INTERVAL_US   1000u

#define LINE_BUF_SIZE            80u

/* Servo */
#define SERVO_ALARM_NUM     2u
#define SERVO_HOME_US       500u
#define SERVO_FIRE_US       2500u
#define SERVO_PERIOD_US     20000u

/* Fire sequence timings */
#define SERVO_EXTEND_MS     400u   /* ms to hold servo extended */
#define SERVO_RETRACT_MS    300u   /* ms to wait after retracting before DONE */
#define INDEXER_DEFAULT_STEPS  800u
#define SERVO_PRE_FIRE_MS    80u   /* brief settle at HOME before indexer starts */

/* =========================================================================
 * Stepper state
 * ========================================================================= */

typedef struct {
    uint step_gpio;
    uint dir_gpio;
    uint slice;
    float step_angle_deg;
    volatile uint32_t target_steps;
    volatile uint32_t completed_steps;
    volatile bool     move_active;
} stepper_t;

static stepper_t g_yaw = {
    YAW_STEP_PIN, YAW_DIR_PIN, YAW_SLICE,
    YAW_STEP_ANGLE_DEG, 0, 0, false
};
static stepper_t g_pitch = {
    PITCH_STEP_PIN, PITCH_DIR_PIN, PITCH_SLICE,
    PITCH_STEP_ANGLE_DEG, 0, 0, false
};
static stepper_t g_indexer = {
    INDEXER_STEP_PIN, INDEXER_DIR_PIN, INDEXER_SLICE,
    INDEXER_STEP_ANGLE_DEG, 0, 0, false
};

/* =========================================================================
 * Yaw velocity ramp state
 * ========================================================================= */

static int32_t g_yaw_vel_target  = 0;
static int32_t g_yaw_vel_current = 0;

/* =========================================================================
 * DShot state
 * ========================================================================= */

static uint16_t g_spin[2] = { 0u, 0u };

/* =========================================================================
 * Servo state (hardware alarm driven)
 * ========================================================================= */

static volatile uint32_t g_servo_pulse_us  = SERVO_HOME_US;
static volatile bool     g_servo_active    = false;
static volatile bool     g_servo_phase_high = false;

/* =========================================================================
 * Fire state machine
 * ========================================================================= */

typedef enum {
    FIRE_IDLE = 0,
    FIRE_INDEXER_MOVING,
    FIRE_SERVO_EXTEND,
    FIRE_SERVO_RETRACT,
} fire_state_t;

static volatile fire_state_t g_fire_state = FIRE_IDLE;
static absolute_time_t       g_fire_timer;

/* =========================================================================
 * Pending DONE notifications
 * ========================================================================= */

static volatile bool g_send_done_yaw   = false;
static volatile bool g_send_done_pitch = false;
static volatile bool g_send_done_feed  = false;

/* =========================================================================
 * USB CDC helpers
 * ========================================================================= */

static void uart_puts_pi(const char *s) { printf("%s", s); }
static void uart_send_ok(void)          { printf("OK\n"); }
static void uart_send_boot(void)        { printf("BOOT\n"); }

/* =========================================================================
 * PWM frequency setup (50% duty, finds smallest integer divider)
 * ========================================================================= */

static bool pwm_set_freq(uint gpio, uint32_t freq_hz) {
    uint32_t sys_clk = clock_get_hz(clk_sys);
    uint32_t divider = (sys_clk / ((uint64_t)freq_hz * 65535u)) + 1u;
    if (divider > 255u) return false;

    uint32_t wrap = sys_clk / (divider * freq_hz);
    if (wrap == 0u) return false;
    if (wrap > 65535u) wrap = 65535u;

    uint slice = pwm_gpio_to_slice_num(gpio);
    uint chan  = pwm_gpio_to_channel(gpio);

    pwm_set_clkdiv_int_frac(slice, (uint8_t)divider, 0u);
    pwm_set_wrap(slice, (uint16_t)(wrap - 1u));
    pwm_set_chan_level(slice, chan, (uint16_t)(wrap / 2u));
    return true;
}

/* =========================================================================
 * Stepper helpers
 * ========================================================================= */

static void stepper_force_idle(stepper_t *m) {
    pwm_set_irq_enabled(m->slice, false);
    pwm_set_enabled(m->slice, false);
    gpio_set_function(m->step_gpio, GPIO_FUNC_SIO);
    gpio_set_dir(m->step_gpio, GPIO_OUT);
    gpio_put(m->step_gpio, false);
    m->move_active = false;
}

static void stepper_init_gpio(stepper_t *m) {
    gpio_init(m->dir_gpio);
    gpio_set_dir(m->dir_gpio, GPIO_OUT);
    gpio_put(m->dir_gpio, false);

    gpio_init(m->step_gpio);
    gpio_set_dir(m->step_gpio, GPIO_OUT);
    gpio_put(m->step_gpio, false);
}

/* Start a move specified in degrees (using per-motor step_angle_deg). */
static bool stepper_start_move(stepper_t *m, float degrees, uint32_t freq_hz) {
    bool forward = degrees >= 0.0f;
    float mag    = forward ? degrees : -degrees;
    uint32_t steps = (uint32_t)(mag / m->step_angle_deg + 0.5f);
    if (steps == 0u) return false;

    gpio_set_function(m->step_gpio, GPIO_FUNC_PWM);
    if (!pwm_set_freq(m->step_gpio, freq_hz)) {
        gpio_set_function(m->step_gpio, GPIO_FUNC_SIO);
        return false;
    }

    gpio_put(m->dir_gpio, forward);
    m->target_steps    = steps;
    m->completed_steps = 0u;
    m->move_active     = true;

    pwm_clear_irq(m->slice);
    pwm_set_irq_enabled(m->slice, true);
    pwm_set_enabled(m->slice, true);
    return true;
}

/* Start a move specified as a raw signed step count. */
static bool stepper_start_move_steps(stepper_t *m, int32_t steps, uint32_t freq_hz) {
    bool forward  = steps >= 0;
    uint32_t abs_steps = (uint32_t)(steps < 0 ? -steps : steps);
    if (abs_steps == 0u) return false;

    gpio_set_function(m->step_gpio, GPIO_FUNC_PWM);
    if (!pwm_set_freq(m->step_gpio, freq_hz)) {
        gpio_set_function(m->step_gpio, GPIO_FUNC_SIO);
        return false;
    }

    gpio_put(m->dir_gpio, forward);
    m->target_steps    = abs_steps;
    m->completed_steps = 0u;
    m->move_active     = true;

    pwm_clear_irq(m->slice);
    pwm_set_irq_enabled(m->slice, true);
    pwm_set_enabled(m->slice, true);
    return true;
}

/* =========================================================================
 * Yaw velocity ramp
 * ========================================================================= */

static void yaw_vel_apply(int32_t freq) {
    if (freq == 0) {
        pwm_set_enabled(g_yaw.slice, false);
        gpio_set_function(g_yaw.step_gpio, GPIO_FUNC_SIO);
        gpio_set_dir(g_yaw.step_gpio, GPIO_OUT);
        gpio_put(g_yaw.step_gpio, false);
    } else {
        bool forward = freq > 0;
        uint32_t abs_freq = (uint32_t)(freq < 0 ? -freq : freq);
        gpio_put(g_yaw.dir_gpio, forward);
        gpio_set_function(g_yaw.step_gpio, GPIO_FUNC_PWM);
        pwm_set_freq(g_yaw.step_gpio, abs_freq);
        pwm_set_enabled(g_yaw.slice, true);
    }
}

static void yaw_vel_ramp_step(void) {
    if (g_yaw.move_active) return;
    if (g_yaw_vel_current == g_yaw_vel_target) return;

    int32_t diff = g_yaw_vel_target - g_yaw_vel_current;
    int32_t step = YAW_VEL_RAMP_HZ_PER_MS;

    if (diff > 0) {
        g_yaw_vel_current = (diff <= step) ? g_yaw_vel_target
                                           : g_yaw_vel_current + step;
    } else {
        g_yaw_vel_current = (-diff <= step) ? g_yaw_vel_target
                                            : g_yaw_vel_current - step;
    }
    yaw_vel_apply(g_yaw_vel_current);
}

static void yaw_vel_stop_immediate(void) {
    g_yaw_vel_target  = 0;
    g_yaw_vel_current = 0;
    yaw_vel_apply(0);
}

/* =========================================================================
 * PWM wrap IRQ — step counter for YAW, PITCH, INDEXER
 * ========================================================================= */

static void pwm_irq_handler(void) {
    uint32_t active = pwm_get_irq_status_mask();

    if (active & (1u << YAW_SLICE)) {
        pwm_clear_irq(YAW_SLICE);
        if (g_yaw.move_active) {
            g_yaw.completed_steps++;
            if (g_yaw.completed_steps >= g_yaw.target_steps) {
                stepper_force_idle(&g_yaw);
                g_send_done_yaw = true;
            }
        }
    }

    if (active & (1u << PITCH_SLICE)) {
        pwm_clear_irq(PITCH_SLICE);
        if (g_pitch.move_active) {
            g_pitch.completed_steps++;
            if (g_pitch.completed_steps >= g_pitch.target_steps) {
                stepper_force_idle(&g_pitch);
                g_send_done_pitch = true;
            }
        }
    }

    if (active & (1u << INDEXER_SLICE)) {
        pwm_clear_irq(INDEXER_SLICE);
        if (g_indexer.move_active) {
            g_indexer.completed_steps++;
            if (g_indexer.completed_steps >= g_indexer.target_steps) {
                stepper_force_idle(&g_indexer);
                /* fire_task() in main loop picks this up and advances state */
            }
        }
    }
}

/* =========================================================================
 * DShot
 * ========================================================================= */

static uint16_t dshot_packet(uint16_t throttle) {
    uint16_t t = throttle > DSHOT_THROTTLE_MAX ? DSHOT_THROTTLE_MAX : throttle;
    uint16_t data = (uint16_t)((t << 1u) & 0xFFFEu);
    uint8_t  crc  = (uint8_t)((data ^ (data >> 4u) ^ (data >> 8u)) & 0x0Fu);
    return (uint16_t)((data << 4u) | crc);
}

static void dshot_send(PIO pio, uint sm, uint16_t throttle) {
    pio_sm_put_blocking(pio, sm, (uint32_t)dshot_packet(throttle) << 16u);
}

static void dshot_init_sm(PIO pio, uint sm, uint pin, uint offset) {
    pio_sm_config cfg = dshot_program_get_default_config(offset);
    float div = (float)clock_get_hz(clk_sys) / ((float)DSHOT_BITRATE_HZ * 12.0f);
    sm_config_set_sideset_pins(&cfg, pin);
    sm_config_set_clkdiv(&cfg, div);
    sm_config_set_out_shift(&cfg, false, true, 16u);
    pio_gpio_init(pio, pin);
    pio_sm_set_consecutive_pindirs(pio, sm, pin, 1u, true);
    pio_sm_init(pio, sm, offset, &cfg);
    pio_sm_set_enabled(pio, sm, true);
}

/* =========================================================================
 * Servo (hardware alarm 2 — independent of PWM)
 * ========================================================================= */

static void servo_alarm_cb(uint alarm_num) {
    (void)alarm_num;
    if (!g_servo_active) {
        gpio_put(SERVO_PIN, false);
        return;
    }
    if (g_servo_phase_high) {
        gpio_put(SERVO_PIN, false);
        g_servo_phase_high = false;
        uint32_t low_us = SERVO_PERIOD_US - g_servo_pulse_us;
        hardware_alarm_set_target(SERVO_ALARM_NUM,
            delayed_by_us(get_absolute_time(), low_us));
    } else {
        gpio_put(SERVO_PIN, true);
        g_servo_phase_high = true;
        hardware_alarm_set_target(SERVO_ALARM_NUM,
            delayed_by_us(get_absolute_time(), g_servo_pulse_us));
    }
}

static void servo_init(void) {
    gpio_init(SERVO_PIN);
    gpio_set_dir(SERVO_PIN, GPIO_OUT);
    gpio_put(SERVO_PIN, false);

    g_servo_pulse_us   = SERVO_HOME_US;
    g_servo_active     = false;
    g_servo_phase_high = false;

    hardware_alarm_claim(SERVO_ALARM_NUM);
    hardware_alarm_set_callback(SERVO_ALARM_NUM, servo_alarm_cb);

    /* Enable and start generating pulses at home position */
    g_servo_active = true;
    hardware_alarm_set_target(SERVO_ALARM_NUM,
        delayed_by_us(get_absolute_time(), 2000u));
}

/* Update servo position — safe to call from anywhere including IRQ context */
static void servo_set(uint32_t pulse_us) {
    if (pulse_us < 500u)  pulse_us = 500u;
    if (pulse_us > 2500u) pulse_us = 2500u;
    g_servo_pulse_us = pulse_us;
}

/* =========================================================================
 * Fire state machine
 * ========================================================================= */

static void fire_task(void) {
    switch (g_fire_state) {
        case FIRE_IDLE:
            return;

        case FIRE_INDEXER_MOVING:
            if (!g_indexer.move_active) {
                /* Indexer done — extend servo */
                servo_set(SERVO_FIRE_US);
                g_fire_timer = delayed_by_ms(get_absolute_time(), SERVO_EXTEND_MS);
                g_fire_state = FIRE_SERVO_EXTEND;
            }
            break;

        case FIRE_SERVO_EXTEND:
            if (absolute_time_diff_us(get_absolute_time(), g_fire_timer) <= 0) {
                servo_set(SERVO_HOME_US);
                g_fire_timer = delayed_by_ms(get_absolute_time(), SERVO_RETRACT_MS);
                g_fire_state = FIRE_SERVO_RETRACT;
            }
            break;

        case FIRE_SERVO_RETRACT:
            if (absolute_time_diff_us(get_absolute_time(), g_fire_timer) <= 0) {
                g_fire_state    = FIRE_IDLE;
                g_send_done_feed = true;
            }
            break;
    }
}

/* =========================================================================
 * Command parser
 * ========================================================================= */

static char *next_token(char **ctx) {
    while (**ctx == ' ' || **ctx == '\t') (*ctx)++;
    if (**ctx == '\0') return NULL;
    char *tok = *ctx;
    while (**ctx && **ctx != ' ' && **ctx != '\t') (*ctx)++;
    if (**ctx) { **ctx = '\0'; (*ctx)++; }
    return tok;
}

static void process_command(char *line) {
    char *ctx = line;
    char *cmd = next_token(&ctx);
    if (!cmd) return;

    /* --- YAW_VEL --- */
    if (strcmp(cmd, "YAW_VEL") == 0) {
        char *a1 = next_token(&ctx);
        if (!a1) { uart_puts_pi("ERR missing_freq\n"); return; }
        int32_t freq = (int32_t)atoi(a1);
        if (g_yaw.move_active) stepper_force_idle(&g_yaw);
        g_yaw_vel_target = freq;
        uart_send_ok();
        return;
    }

    /* --- YAW / PITCH (degree-based) --- */
    bool is_yaw   = strcmp(cmd, "YAW")   == 0;
    bool is_pitch = strcmp(cmd, "PITCH") == 0;

    if (is_yaw || is_pitch) {
        stepper_t  *motor = is_yaw ? &g_yaw : &g_pitch;
        const char *name  = is_yaw ? "YAW" : "PITCH";

        if (is_yaw) yaw_vel_stop_immediate();

        if (motor->move_active) {
            char buf[24];
            snprintf(buf, sizeof(buf), "BUSY %s\n", name);
            uart_puts_pi(buf);
            return;
        }

        char *deg_s  = next_token(&ctx);
        char *freq_s = next_token(&ctx);
        if (!deg_s) { uart_puts_pi("ERR missing_angle\n"); return; }

        float    deg  = strtof(deg_s, NULL);
        uint32_t freq = freq_s ? (uint32_t)strtoul(freq_s, NULL, 10) : STEPPER_DEFAULT_FREQ_HZ;
        if (freq == 0u) freq = STEPPER_DEFAULT_FREQ_HZ;

        if (!stepper_start_move(motor, deg, freq)) {
            uart_puts_pi("ERR stepper_config_failed\n");
            return;
        }
        uart_send_ok();
        return;
    }

    /* --- PITCH_STEPS <n> [freq_hz] --- */
    if (strcmp(cmd, "PITCH_STEPS") == 0) {
        if (g_pitch.move_active) { uart_puts_pi("BUSY PITCH\n"); return; }

        char *steps_s = next_token(&ctx);
        char *freq_s  = next_token(&ctx);
        if (!steps_s) { uart_puts_pi("ERR missing_steps\n"); return; }

        int32_t  steps = (int32_t)strtol(steps_s, NULL, 10);
        uint32_t freq  = freq_s ? (uint32_t)strtoul(freq_s, NULL, 10) : STEPPER_DEFAULT_FREQ_HZ;
        if (freq == 0u) freq = STEPPER_DEFAULT_FREQ_HZ;

        if (!stepper_start_move_steps(&g_pitch, steps, freq)) {
            uart_puts_pi("ERR stepper_config_failed\n");
            return;
        }
        uart_send_ok();
        return;
    }

    /* --- FIRE <steps> [freq_hz] ---
     * Ensure servo is at home, start indexer, then servo extend/retract. */
    if (strcmp(cmd, "FIRE") == 0) {
        if (g_fire_state != FIRE_IDLE) { uart_puts_pi("BUSY FEED\n"); return; }

        char *steps_s = next_token(&ctx);
        char *freq_s  = next_token(&ctx);
        int32_t  steps = steps_s ? (int32_t)strtol(steps_s, NULL, 10)
                                 : (int32_t)INDEXER_DEFAULT_STEPS;
        uint32_t freq  = freq_s ? (uint32_t)strtoul(freq_s, NULL, 10) : STEPPER_DEFAULT_FREQ_HZ;
        if (steps <= 0)  steps = (int32_t)INDEXER_DEFAULT_STEPS;
        if (freq == 0u)  freq  = STEPPER_DEFAULT_FREQ_HZ;

        /* Servo must be at HOME before indexer can move */
        servo_set(SERVO_HOME_US);
        sleep_ms(SERVO_PRE_FIRE_MS);

        if (!stepper_start_move_steps(&g_indexer, steps, freq)) {
            uart_puts_pi("ERR indexer_config_failed\n");
            return;
        }
        g_fire_state = FIRE_INDEXER_MOVING;
        uart_send_ok();
        return;
    }

    /* --- FEED [counts] — backward-compat alias for FIRE --- */
    if (strcmp(cmd, "FEED") == 0) {
        if (g_fire_state != FIRE_IDLE) { uart_puts_pi("BUSY FEED\n"); return; }
        servo_set(SERVO_HOME_US);
        sleep_ms(SERVO_PRE_FIRE_MS);
        if (!stepper_start_move_steps(&g_indexer, (int32_t)INDEXER_DEFAULT_STEPS,
                                      STEPPER_DEFAULT_FREQ_HZ)) {
            uart_puts_pi("ERR indexer_config_failed\n");
            return;
        }
        g_fire_state = FIRE_INDEXER_MOVING;
        uart_send_ok();
        return;
    }

    /* --- FEED_MS <ms> — backward-compat alias for FIRE --- */
    if (strcmp(cmd, "FEED_MS") == 0) {
        if (g_fire_state != FIRE_IDLE) { uart_puts_pi("BUSY FEED\n"); return; }
        servo_set(SERVO_HOME_US);
        sleep_ms(SERVO_PRE_FIRE_MS);
        if (!stepper_start_move_steps(&g_indexer, (int32_t)INDEXER_DEFAULT_STEPS,
                                      STEPPER_DEFAULT_FREQ_HZ)) {
            uart_puts_pi("ERR indexer_config_failed\n");
            return;
        }
        g_fire_state = FIRE_INDEXER_MOVING;
        uart_send_ok();
        return;
    }

    /* --- SPIN --- */
    if (strcmp(cmd, "SPIN") == 0) {
        char *t1s = next_token(&ctx);
        char *t2s = next_token(&ctx);
        if (!t1s || !t2s) { uart_puts_pi("ERR missing_throttle\n"); return; }
        uint32_t t1 = strtoul(t1s, NULL, 10);
        uint32_t t2 = strtoul(t2s, NULL, 10);
        if (t1 > DSHOT_THROTTLE_MAX) t1 = DSHOT_THROTTLE_MAX;
        if (t2 > DSHOT_THROTTLE_MAX) t2 = DSHOT_THROTTLE_MAX;
        g_spin[0] = (uint16_t)t1;
        g_spin[1] = (uint16_t)t2;
        uart_send_ok();
        return;
    }

    /* --- HOME --- */
    if (strcmp(cmd, "HOME") == 0) {
        g_yaw.completed_steps     = 0u;
        g_yaw.target_steps        = 0u;
        g_pitch.completed_steps   = 0u;
        g_pitch.target_steps      = 0u;
        g_indexer.completed_steps = 0u;
        g_indexer.target_steps    = 0u;
        uart_send_ok();
        return;
    }

    /* --- STATUS --- */
    if (strcmp(cmd, "STATUS") == 0) {
        char buf[96];
        snprintf(buf, sizeof(buf),
            "STATUS yaw_moving=%u pitch_moving=%u indexer_moving=%u fire_active=%u spin1=%u spin2=%u\n",
            (unsigned)g_yaw.move_active,
            (unsigned)g_pitch.move_active,
            (unsigned)g_indexer.move_active,
            (unsigned)(g_fire_state != FIRE_IDLE),
            (unsigned)g_spin[0],
            (unsigned)g_spin[1]);
        uart_puts_pi(buf);
        return;
    }

    /* --- ESTOP --- */
    if (strcmp(cmd, "ESTOP") == 0) {
        yaw_vel_stop_immediate();
        stepper_force_idle(&g_yaw);
        stepper_force_idle(&g_pitch);
        stepper_force_idle(&g_indexer);
        g_fire_state = FIRE_IDLE;
        servo_set(SERVO_HOME_US);
        g_spin[0] = 0u;
        g_spin[1] = 0u;
        g_send_done_yaw   = false;
        g_send_done_pitch = false;
        g_send_done_feed  = false;
        uart_send_ok();
        return;
    }

    uart_puts_pi("ERR unknown_command\n");
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void) {
    stdio_init_all();

    while (!stdio_usb_connected()) sleep_ms(10);
    sleep_ms(100);

    /* Servo must be at home before anything else can move */
    servo_init();

    /* Steppers */
    stepper_init_gpio(&g_yaw);
    stepper_init_gpio(&g_pitch);
    stepper_init_gpio(&g_indexer);
    irq_set_exclusive_handler(PWM_DEFAULT_IRQ_NUM(), pwm_irq_handler);
    irq_set_enabled(PWM_DEFAULT_IRQ_NUM(), true);

    /* DShot flywheels */
    PIO pio = pio0;
    uint offset = pio_add_program(pio, &dshot_program);
    dshot_init_sm(pio, 0u, DSHOT_PIN_1, offset);
    dshot_init_sm(pio, 1u, DSHOT_PIN_2, offset);

    uart_send_boot();

    char     line[LINE_BUF_SIZE];
    uint32_t line_len = 0u;

    absolute_time_t next_dshot        = get_absolute_time();
    absolute_time_t next_yaw_vel_ramp = get_absolute_time();

    while (true) {
        /* Read USB CDC (non-blocking) */
        int raw;
        while ((raw = getchar_timeout_us(0)) != PICO_ERROR_TIMEOUT) {
            char ch = (char)raw;
            if (ch == '\n' || ch == '\r') {
                if (line_len > 0u) {
                    line[line_len] = '\0';
                    process_command(line);
                    line_len = 0u;
                }
            } else if (line_len < LINE_BUF_SIZE - 1u) {
                line[line_len++] = ch;
            } else {
                line_len = 0u;
                uart_puts_pi("ERR line_too_long\n");
            }
        }

        /* Yaw velocity ramp at 1 ms */
        if (absolute_time_diff_us(get_absolute_time(), next_yaw_vel_ramp) <= 0) {
            yaw_vel_ramp_step();
            next_yaw_vel_ramp = delayed_by_us(next_yaw_vel_ramp, 1000u);
        }

        /* DShot at 1 ms */
        if (absolute_time_diff_us(get_absolute_time(), next_dshot) <= 0) {
            dshot_send(pio, 0u, g_spin[0]);
            dshot_send(pio, 1u, g_spin[1]);
            next_dshot = delayed_by_us(next_dshot, DSHOT_SEND_INTERVAL_US);
        }

        /* Fire state machine */
        fire_task();

        /* Proactive DONE notifications */
        if (g_send_done_yaw) {
            g_send_done_yaw = false;
            uart_puts_pi("DONE YAW\n");
        }
        if (g_send_done_pitch) {
            g_send_done_pitch = false;
            uart_puts_pi("DONE PITCH\n");
        }
        if (g_send_done_feed) {
            g_send_done_feed = false;
            uart_puts_pi("DONE FEED\n");
        }

        tight_loop_contents();
    }
}
