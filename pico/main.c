/**
 * Beer Pong Turret — Pico Firmware
 *
 * Pin assignments (match existing hardware wiring):
 *   USB CDC (Pi comms): USB micro-B on Pico → USB-A on Pi (/dev/ttyACM0)
 *   Stepper YAW    : STEP=GPIO27, DIR=GPIO26
 *   Stepper PITCH  : STEP=GPIO17, DIR=GPIO16
 *   DShot Motor 1  : GPIO12
 *   DShot Motor 2  : GPIO13
 *   Feed Motor PWM : GPIO14
 *   Feed Motor DIR : GPIO15
 *   Feed Encoder A : GPIO18  (rising-edge interrupt)
 *
 * Protocol: text lines, newline-terminated.
 *
 * Pi → Pico commands:
 *   YAW <deg> [freq_hz]       move yaw stepper
 *   PITCH <deg> [freq_hz]     move pitch stepper
 *   SPIN <t1> <t2>            set flywheel DShot throttle (0-2047 each)
 *   FEED [counts]             run feed motor N encoder counts (default 200)
 *   HOME                      zero position counters, no motion
 *   STATUS                    query state
 *   ESTOP                     stop everything
 *
 * Pico → Pi responses/events:
 *   OK
 *   BUSY YAW | BUSY PITCH | BUSY FEED
 *   DONE YAW                  proactively sent when yaw move finishes
 *   DONE PITCH                proactively sent when pitch move finishes
 *   DONE FEED                 proactively sent when feed finishes
 *   STATUS yaw_moving=0 pitch_moving=0 feed_active=0 spin1=0 spin2=0
 *   ERR <reason>
 *   BOOT                      sent once on startup
 */

#include "pico/stdlib.h"
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "hardware/pio.h"
#include "hardware/clocks.h"
#include "hardware/irq.h"
#include "dshot.pio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* =========================================================================
 * Pin definitions
 * ========================================================================= */

/* Communication: USB CDC — no UART pin defines needed.
 * Pico appears as /dev/ttyACM0 (or ttyACM1) on the Raspberry Pi. */

#define YAW_STEP_PIN        27u
#define YAW_DIR_PIN         26u

#define PITCH_STEP_PIN      17u
#define PITCH_DIR_PIN       16u

#define DSHOT_PIN_1         12u
#define DSHOT_PIN_2         13u

#define FEED_PWM_PIN        14u
#define FEED_DIR_PIN        15u
#define FEED_ENC_PIN        18u

/* =========================================================================
 * Derived PWM slices
 *   slice = (gpio / 2) % 8
 * ========================================================================= */

#define YAW_SLICE           5u   /* GPIO27: (27/2)%8 = 5 */
#define PITCH_SLICE         0u   /* GPIO17: (17/2)%8 = 0 */
#define FEED_SLICE          7u   /* GPIO14: (14/2)%8 = 7 */

/* =========================================================================
 * Tuning constants
 * ========================================================================= */

#define STEPPER_STEP_ANGLE_DEG   1.8f   /* degrees per full step */
#define STEPPER_DEFAULT_FREQ_HZ  500u
#define DSHOT_BITRATE_HZ         150000u
#define DSHOT_THROTTLE_MAX       2047u
#define DSHOT_SEND_INTERVAL_US   1000u   /* 1 ms */
#define FEED_DEFAULT_COUNTS      200u    /* encoder counts per ball feed */
#define FEED_PWM_FREQ_HZ         20000u  /* 20 kHz feed motor PWM */
#define FEED_DUTY_PCT            80u     /* feed motor duty while active (%) */
#define LINE_BUF_SIZE            80u

/* =========================================================================
 * Stepper state
 * ========================================================================= */

typedef struct {
    uint step_gpio;
    uint dir_gpio;
    uint slice;
    volatile uint32_t target_steps;
    volatile uint32_t completed_steps;
    volatile bool     move_active;
} stepper_t;

static stepper_t g_yaw   = { YAW_STEP_PIN,   YAW_DIR_PIN,   YAW_SLICE,   0, 0, false };
static stepper_t g_pitch = { PITCH_STEP_PIN, PITCH_DIR_PIN, PITCH_SLICE, 0, 0, false };

/* =========================================================================
 * DShot state
 * ========================================================================= */

static uint16_t g_spin[2] = { 0u, 0u };

/* =========================================================================
 * Feed motor state
 * ========================================================================= */

static volatile int32_t  g_feed_enc_count  = 0;
static volatile int32_t  g_feed_target     = 0;
static volatile bool     g_feed_active     = false;

/* =========================================================================
 * Pending "DONE" notifications to send from main loop
 * ========================================================================= */

static volatile bool g_send_done_yaw   = false;
static volatile bool g_send_done_pitch = false;
static volatile bool g_send_done_feed  = false;

/* =========================================================================
 * USB CDC output helpers
 * ========================================================================= */

static void uart_puts_pi(const char *s) {
    /* printf goes to USB CDC when stdio_init_all() has been called */
    printf("%s", s);
}

static void uart_send_ok(void)   { printf("OK\n"); }
static void uart_send_boot(void) { printf("BOOT\n"); }

/* =========================================================================
 * PWM frequency setup
 *   Finds the smallest integer divider (1-255) that gives wrap <= 65535.
 *   Sets 50% duty cycle.
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
    pwm_set_chan_level(slice, chan, (uint16_t)(wrap / 2u));  /* 50% duty */
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

/**
 * Start a stepper move.
 * degrees > 0 → forward direction, degrees < 0 → reverse.
 * Returns false if steps=0 or PWM config fails.
 */
static bool stepper_start_move(stepper_t *m, float degrees, uint32_t freq_hz) {
    bool forward = degrees >= 0.0f;
    float mag    = forward ? degrees : -degrees;
    uint32_t steps = (uint32_t)(mag / STEPPER_STEP_ANGLE_DEG + 0.5f);
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

/* =========================================================================
 * PWM wrap IRQ — step counter for YAW and PITCH only.
 * FEED slice (7) is intentionally excluded.
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
}

/* =========================================================================
 * DShot
 * ========================================================================= */

static uint16_t dshot_packet(uint16_t throttle) {
    uint16_t t = throttle > DSHOT_THROTTLE_MAX ? DSHOT_THROTTLE_MAX : throttle;
    uint16_t data = (uint16_t)((t << 1u) & 0xFFFEu);  /* telemetry=0 */
    uint8_t crc = (uint8_t)((data ^ (data >> 4u) ^ (data >> 8u)) & 0x0Fu);
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
 * Feed motor
 * ========================================================================= */

static void feed_stop(void) {
    pwm_set_chan_level(FEED_SLICE, PWM_CHAN_A, 0u);
    g_feed_active = false;
}

/** GPIO interrupt for feed encoder — counts rising edges. */
static void feed_enc_irq(uint gpio, uint32_t events) {
    (void)gpio; (void)events;
    if (!g_feed_active) return;
    g_feed_enc_count++;
    if (g_feed_enc_count >= g_feed_target) {
        feed_stop();
        g_send_done_feed = true;
    }
}

static void feed_init(void) {
    /* Direction pin */
    gpio_init(FEED_DIR_PIN);
    gpio_set_dir(FEED_DIR_PIN, GPIO_OUT);
    gpio_put(FEED_DIR_PIN, true);

    /* PWM output */
    gpio_set_function(FEED_PWM_PIN, GPIO_FUNC_PWM);
    pwm_set_freq(FEED_PWM_PIN, FEED_PWM_FREQ_HZ);
    pwm_set_chan_level(FEED_SLICE, PWM_CHAN_A, 0u);  /* start stopped */
    pwm_set_enabled(FEED_SLICE, true);

    /* Encoder input */
    gpio_init(FEED_ENC_PIN);
    gpio_set_dir(FEED_ENC_PIN, GPIO_IN);
    gpio_pull_up(FEED_ENC_PIN);
    gpio_set_irq_enabled_with_callback(FEED_ENC_PIN, GPIO_IRQ_EDGE_RISE, true, feed_enc_irq);
}

static void feed_start(int32_t counts) {
    /* Compute 80% duty cycle level from current wrap */
    uint32_t wrap = pwm_hw->slice[FEED_SLICE].top + 1u;
    uint32_t level = wrap * FEED_DUTY_PCT / 100u;

    g_feed_enc_count = 0;
    g_feed_target    = counts;
    g_feed_active    = true;
    pwm_set_chan_level(FEED_SLICE, PWM_CHAN_A, (uint16_t)level);
}

/* =========================================================================
 * Command parser
 * ========================================================================= */

/** Advance past spaces, null-terminate the next token, return pointer to it. */
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

    /* --- YAW / PITCH --- */
    bool is_yaw   = strcmp(cmd, "YAW")   == 0;
    bool is_pitch = strcmp(cmd, "PITCH") == 0;

    if (is_yaw || is_pitch) {
        stepper_t *motor = is_yaw ? &g_yaw : &g_pitch;
        const char *name = is_yaw ? "YAW" : "PITCH";

        if (motor->move_active) {
            char buf[24];
            snprintf(buf, sizeof(buf), "BUSY %s\n", name);
            uart_puts_pi(buf);
            return;
        }

        char *deg_s  = next_token(&ctx);
        char *freq_s = next_token(&ctx);
        if (!deg_s) { uart_puts_pi("ERR missing_angle\n"); return; }

        float deg = strtof(deg_s, NULL);
        uint32_t freq = freq_s ? (uint32_t)strtoul(freq_s, NULL, 10) : STEPPER_DEFAULT_FREQ_HZ;
        if (freq == 0u) freq = STEPPER_DEFAULT_FREQ_HZ;

        if (!stepper_start_move(motor, deg, freq)) {
            uart_puts_pi("ERR stepper_config_failed\n");
            return;
        }
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

    /* --- FEED --- */
    if (strcmp(cmd, "FEED") == 0) {
        if (g_feed_active) { uart_puts_pi("BUSY FEED\n"); return; }
        char *cnt_s = next_token(&ctx);
        int32_t counts = cnt_s ? (int32_t)strtoul(cnt_s, NULL, 10) : (int32_t)FEED_DEFAULT_COUNTS;
        if (counts <= 0) counts = (int32_t)FEED_DEFAULT_COUNTS;
        feed_start(counts);
        uart_send_ok();
        return;
    }

    /* --- HOME --- */
    if (strcmp(cmd, "HOME") == 0) {
        /* Zero counters — operator is responsible for physical position */
        g_yaw.completed_steps   = 0u;
        g_yaw.target_steps      = 0u;
        g_pitch.completed_steps = 0u;
        g_pitch.target_steps    = 0u;
        uart_send_ok();
        return;
    }

    /* --- STATUS --- */
    if (strcmp(cmd, "STATUS") == 0) {
        char buf[80];
        snprintf(buf, sizeof(buf),
            "STATUS yaw_moving=%u pitch_moving=%u feed_active=%u spin1=%u spin2=%u\n",
            (unsigned)g_yaw.move_active,
            (unsigned)g_pitch.move_active,
            (unsigned)g_feed_active,
            (unsigned)g_spin[0],
            (unsigned)g_spin[1]);
        uart_puts_pi(buf);
        return;
    }

    /* --- ESTOP --- */
    if (strcmp(cmd, "ESTOP") == 0) {
        stepper_force_idle(&g_yaw);
        stepper_force_idle(&g_pitch);
        feed_stop();
        g_spin[0] = 0u;
        g_spin[1] = 0u;
        /* Clear any pending DONE that would be confusing after estop */
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
    stdio_init_all();   /* enables USB CDC serial */

    /* Wait for USB CDC host connection before sending anything.
     * On the Pi, pyserial opening /dev/ttyACM0 triggers this. */
    while (!stdio_usb_connected()) {
        sleep_ms(10);
    }
    sleep_ms(100);      /* brief settle after connect */
    uart_send_boot();

    /* Steppers */
    stepper_init_gpio(&g_yaw);
    stepper_init_gpio(&g_pitch);
    irq_set_exclusive_handler(PWM_DEFAULT_IRQ_NUM(), pwm_irq_handler);
    irq_set_enabled(PWM_DEFAULT_IRQ_NUM(), true);

    /* DShot flywheels */
    PIO pio = pio0;
    uint offset = pio_add_program(pio, &dshot_program);
    dshot_init_sm(pio, 0u, DSHOT_PIN_1, offset);
    dshot_init_sm(pio, 1u, DSHOT_PIN_2, offset);

    /* Feed motor */
    feed_init();

    /* Line buffer for incoming commands */
    char line[LINE_BUF_SIZE];
    uint32_t line_len = 0u;

    absolute_time_t next_dshot = get_absolute_time();

    while (true) {
        /* ---- Read USB CDC (non-blocking) ---- */
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
                /* Buffer overflow — discard */
                line_len = 0u;
                uart_puts_pi("ERR line_too_long\n");
            }
        }

        /* ---- DShot at 1 ms interval ---- */
        if (absolute_time_diff_us(get_absolute_time(), next_dshot) <= 0) {
            dshot_send(pio, 0u, g_spin[0]);
            dshot_send(pio, 1u, g_spin[1]);
            next_dshot = delayed_by_us(next_dshot, DSHOT_SEND_INTERVAL_US);
        }

        /* ---- Proactive DONE notifications ---- */
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
