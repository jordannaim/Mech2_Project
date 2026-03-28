#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/pwm.h"
#include "pico/stdlib.h"
#include "dshot.pio.h"
#include "pwm_generator.h"
#include "pwm_generator_pico.h"

#define DSHOT_MOTOR_COUNT 2u
#define DSHOT_GPIO_MOTOR_1 12u
#define DSHOT_GPIO_MOTOR_2 13u
#define DSHOT_BITRATE 150000u
#define DSHOT_THROTTLE_MAX 2047u
#define DSHOT_DEFAULT_THROTTLE 0u
#define STEPPER_MOTOR_COUNT 2u
#define STEPPER_1_STEP_GPIO 27u
#define STEPPER_1_DIR_GPIO 26u
#define STEPPER_2_STEP_GPIO 17u
#define STEPPER_2_DIR_GPIO 16u
#define STEPPER_DEFAULT_FREQUENCY_HZ 500u
#define STEPPER_DEFAULT_DUTY_CYCLE 0.50f
#define STEPPER_STEP_ANGLE_DEG 1.8
#define SERIAL_LINE_BUFFER_SIZE 64u
#define DSHOT_SEND_INTERVAL_US 1000u

typedef struct {
    uint32_t step_frequency_hz;
    pwm_generator_config_t pwm_config;
    volatile uint32_t target_steps;
    volatile uint32_t completed_steps;
    volatile bool move_active;
    bool direction_forward;
} stepper_motion_t;

typedef struct {
    uint step_gpio;
    uint dir_gpio;
    uint pwm_slice;
    stepper_motion_t motion;
} stepper_motor_t;

static stepper_motor_t g_stepper_motors[STEPPER_MOTOR_COUNT] = {
    {
        .step_gpio = STEPPER_1_STEP_GPIO,
        .dir_gpio = STEPPER_1_DIR_GPIO,
        .pwm_slice = 0u,
        .motion =
            {
                .step_frequency_hz = STEPPER_DEFAULT_FREQUENCY_HZ,
                .target_steps = 0u,
                .completed_steps = 0u,
                .move_active = false,
                .direction_forward = true,
            },
    },
    {
        .step_gpio = STEPPER_2_STEP_GPIO,
        .dir_gpio = STEPPER_2_DIR_GPIO,
        .pwm_slice = 0u,
        .motion =
            {
                .step_frequency_hz = STEPPER_DEFAULT_FREQUENCY_HZ,
                .target_steps = 0u,
                .completed_steps = 0u,
                .move_active = false,
                .direction_forward = true,
            },
    },
};

static const uint g_dshot_gpios[DSHOT_MOTOR_COUNT] = {
    DSHOT_GPIO_MOTOR_1,
    DSHOT_GPIO_MOTOR_2,
};

static uint8_t dshot_calculate_crc(uint16_t data) {
    uint8_t checksum = 0u;

    checksum ^= (uint8_t)(data & 0x0Fu);
    checksum ^= (uint8_t)((data >> 4u) & 0x0Fu);
    checksum ^= (uint8_t)((data >> 8u) & 0x0Fu);

    return (uint8_t)(checksum & 0x0Fu);
}

static uint16_t dshot_create_packet(uint16_t throttle, bool telemetry) {
    const uint16_t clamped_throttle =
        throttle > DSHOT_THROTTLE_MAX ? DSHOT_THROTTLE_MAX : throttle;
    const uint16_t data = (uint16_t)((clamped_throttle << 1u) | (telemetry ? 1u : 0u));
    const uint8_t crc = dshot_calculate_crc(data);

    return (uint16_t)((data << 4u) | crc);
}

static void dshot_init(PIO pio, uint sm, uint pin, uint offset, uint32_t bitrate_hz) {
    pio_sm_config config = dshot_program_get_default_config(offset);
    const float divider = (float)clock_get_hz(clk_sys) / ((float)bitrate_hz * 12.0f);

    sm_config_set_sideset_pins(&config, pin);
    sm_config_set_clkdiv(&config, divider);
    sm_config_set_out_shift(&config, false, true, 16u);

    pio_gpio_init(pio, pin);
    pio_sm_set_consecutive_pindirs(pio, sm, pin, 1, true);
    pio_sm_init(pio, sm, offset, &config);
    pio_sm_set_enabled(pio, sm, true);
}

static void dshot_send(PIO pio, uint sm, uint16_t throttle) {
    const uint16_t packet = dshot_create_packet(throttle, false);
    const uint32_t tx_word = ((uint32_t)packet) << 16u;

    pio_sm_put_blocking(pio, sm, tx_word);
}

static bool stepper_apply_frequency(uint step_gpio,
                                    uint32_t frequency_hz,
                                    float duty_cycle,
                                    pwm_generator_config_t *config) {
    if (pwm_generator_calculate(clock_get_hz(clk_sys), frequency_hz, duty_cycle, config) !=
        PWM_GENERATOR_OK) {
        return false;
    }

    return pwm_generator_apply_gpio(step_gpio, config);
}

static void stepper_force_idle_output(stepper_motor_t *motor) {
    pwm_set_irq_enabled(motor->pwm_slice, false);
    pwm_set_enabled(motor->pwm_slice, false);
    gpio_set_function(motor->step_gpio, GPIO_FUNC_SIO);
    gpio_set_dir(motor->step_gpio, GPIO_OUT);
    gpio_put(motor->step_gpio, false);
}

static void stepper_pwm_wrap_irq_handler(void) {
    const uint32_t active_irqs = pwm_get_irq_status_mask();

    for (uint motor_index = 0u; motor_index < STEPPER_MOTOR_COUNT; ++motor_index) {
        stepper_motor_t *motor = &g_stepper_motors[motor_index];

        if ((active_irqs & (1u << motor->pwm_slice)) == 0u) {
            continue;
        }

        pwm_clear_irq(motor->pwm_slice);

        if (!motor->motion.move_active) {
            continue;
        }

        ++motor->motion.completed_steps;
        if (motor->motion.completed_steps >= motor->motion.target_steps) {
            motor->motion.move_active = false;
            stepper_force_idle_output(motor);
        }
    }
}

static void stepper_init(void) {
    for (uint motor_index = 0u; motor_index < STEPPER_MOTOR_COUNT; ++motor_index) {
        stepper_motor_t *motor = &g_stepper_motors[motor_index];
        motor->pwm_slice = pwm_gpio_to_slice_num(motor->step_gpio);

        gpio_init(motor->dir_gpio);
        gpio_set_dir(motor->dir_gpio, GPIO_OUT);
        gpio_put(motor->dir_gpio, false);

        gpio_init(motor->step_gpio);
        gpio_set_dir(motor->step_gpio, GPIO_OUT);
        gpio_put(motor->step_gpio, false);
    }

    irq_set_exclusive_handler(PWM_DEFAULT_IRQ_NUM(), stepper_pwm_wrap_irq_handler);
    irq_set_enabled(PWM_DEFAULT_IRQ_NUM(), true);
}

static bool stepper_start_move(double requested_degrees,
                               uint32_t requested_frequency_hz,
                               stepper_motor_t *motor) {
    const bool direction_forward = requested_degrees >= 0.0;
    const double magnitude_degrees = direction_forward ? requested_degrees : -requested_degrees;
    const uint32_t requested_steps =
        (uint32_t)((magnitude_degrees / STEPPER_STEP_ANGLE_DEG) + 0.5);

    if (requested_steps == 0u) {
        return false;
    }

    if (!stepper_apply_frequency(
            motor->step_gpio, requested_frequency_hz, STEPPER_DEFAULT_DUTY_CYCLE, &motor->motion.pwm_config)) {
        return false;
    }

    pwm_set_enabled(motor->pwm_slice, false);
    gpio_put(motor->dir_gpio, direction_forward);

    motor->motion.step_frequency_hz = requested_frequency_hz;
    motor->motion.target_steps = requested_steps;
    motor->motion.completed_steps = 0u;
    motor->motion.direction_forward = direction_forward;
    motor->motion.move_active = true;

    pwm_clear_irq(motor->pwm_slice);
    pwm_set_irq_enabled(motor->pwm_slice, true);
    pwm_set_enabled(motor->pwm_slice, true);

    return true;
}

static void print_status(const stepper_motor_t steppers[STEPPER_MOTOR_COUNT],
                         const uint16_t dshot_throttles[DSHOT_MOTOR_COUNT]) {
    for (uint motor_index = 0u; motor_index < STEPPER_MOTOR_COUNT; ++motor_index) {
        const stepper_motor_t *motor = &steppers[motor_index];
        printf("Stepper motor %u STEP GPIO %u DIR GPIO %u: target=%lu Hz achieved=%lu Hz duty=%.1f%% move=%s %lu/%lu steps\n",
               motor_index + 1u,
               motor->step_gpio,
               motor->dir_gpio,
               (unsigned long)motor->motion.step_frequency_hz,
               (unsigned long)motor->motion.pwm_config.achieved_frequency_hz,
               STEPPER_DEFAULT_DUTY_CYCLE * 100.0f,
               motor->motion.move_active ? "active" : "idle",
               (unsigned long)motor->motion.completed_steps,
               (unsigned long)motor->motion.target_steps);
    }
    printf("DSHOT motor 1 GPIO %u: constant throttle=%u bitrate=%lu\n",
           g_dshot_gpios[0],
           dshot_throttles[0],
           (unsigned long)DSHOT_BITRATE);
    printf("DSHOT motor 2 GPIO %u: constant throttle=%u bitrate=%lu\n",
           g_dshot_gpios[1],
           dshot_throttles[1],
           (unsigned long)DSHOT_BITRATE);
}

static void print_help(void) {
    printf("Commands:\n");
    printf("  status                          Show current stepper and DSHOT settings\n");
    printf("  dshot <value>                   Set both DSHOT motors to one throttle value\n");
    printf("  dshot <motor> <value>           Set one DSHOT motor (motor 1 or 2)\n");
    printf("  dshot <value1> <value2>         Set motor 1 and motor 2 independently\n");
    printf("  step <degrees> [freq_hz]        Move stepper motor 1 using GPIO %u/%u\n",
           STEPPER_1_STEP_GPIO,
           STEPPER_1_DIR_GPIO);
    printf("  step <motor> <degrees> [freq_hz] Move stepper motor 1 or 2\n");
    printf("  stop                            Stop any active move on stepper motor 1\n");
    printf("  stop <motor>                    Stop any active move on stepper motor 1 or 2\n");
    printf("  help                            Show this help\n");
}

static char *next_token(char **input) {
    char *token_start = *input;

    while (*token_start == ' ' || *token_start == '\t') {
        ++token_start;
    }

    if (*token_start == '\0') {
        *input = token_start;
        return NULL;
    }

    char *token_end = token_start;
    while (*token_end != '\0' && *token_end != ' ' && *token_end != '\t') {
        ++token_end;
    }

    if (*token_end != '\0') {
        *token_end = '\0';
        *input = token_end + 1;
    } else {
        *input = token_end;
    }

    return token_start;
}

static bool parse_stepper_index(const char *text, uint *motor_index) {
    char *end = NULL;
    const unsigned long parsed_motor = strtoul(text, &end, 10);

    if (end == text || *end != '\0' || parsed_motor == 0u || parsed_motor > STEPPER_MOTOR_COUNT) {
        return false;
    }

    *motor_index = (uint)parsed_motor - 1u;
    return true;
}

static void process_serial_command(char *line,
                                   stepper_motor_t steppers[STEPPER_MOTOR_COUNT],
                                   uint16_t dshot_throttles[DSHOT_MOTOR_COUNT]) {
    char *context = line;
    char *command = next_token(&context);

    if (command == NULL) {
        return;
    }

    if (strcmp(command, "status") == 0) {
        print_status(steppers, dshot_throttles);
        return;
    }

    if (strcmp(command, "help") == 0) {
        print_help();
        return;
    }

    if (strcmp(command, "dshot") == 0) {
        char *first_value_text = next_token(&context);
        char *second_value_text = next_token(&context);

        if (first_value_text == NULL) {
            printf("Missing DSHOT throttle\n");
            return;
        }

        char *first_end = NULL;
        const unsigned long first_value = strtoul(first_value_text, &first_end, 10);
        if (first_end == first_value_text || *first_end != '\0' || first_value > DSHOT_THROTTLE_MAX) {
            printf("Invalid DSHOT value: %s\n", first_value_text);
            return;
        }

        if (second_value_text == NULL) {
            dshot_throttles[0] = (uint16_t)first_value;
            dshot_throttles[1] = (uint16_t)first_value;
            printf("DSHOT motor 1 and motor 2 throttle updated: %u\n", (uint16_t)first_value);
            return;
        }

        char *second_end = NULL;
        const unsigned long second_value = strtoul(second_value_text, &second_end, 10);
        if (second_end == second_value_text || *second_end != '\0' || second_value > DSHOT_THROTTLE_MAX) {
            printf("Invalid DSHOT value: %s\n", second_value_text);
            return;
        }

        if ((first_value == 1u || first_value == 2u) && next_token(&context) == NULL) {
            const uint motor_index = (uint)first_value - 1u;
            dshot_throttles[motor_index] = (uint16_t)second_value;
            printf("DSHOT motor %lu throttle updated: %u\n",
                   first_value,
                   dshot_throttles[motor_index]);
            return;
        }

        dshot_throttles[0] = (uint16_t)first_value;
        dshot_throttles[1] = (uint16_t)second_value;
        printf("DSHOT motor 1 throttle updated: %u\n", dshot_throttles[0]);
        printf("DSHOT motor 2 throttle updated: %u\n", dshot_throttles[1]);
        return;
    }

    if (strcmp(command, "step") == 0) {
        char *first_arg = next_token(&context);
        char *second_arg = next_token(&context);
        char *third_arg = next_token(&context);
        uint motor_index = 0u;
        char *degrees_text = first_arg;
        char *frequency_text = second_arg;

        if (first_arg == NULL) {
            printf("Missing stepper angle\n");
            return;
        }

        if (second_arg != NULL && parse_stepper_index(first_arg, &motor_index)) {
            degrees_text = second_arg;
            frequency_text = third_arg;
        } else if (third_arg != NULL) {
            printf("Too many step arguments\n");
            return;
        }

        char *degrees_end = NULL;
        const double requested_degrees = strtod(degrees_text, &degrees_end);
        if (degrees_end == degrees_text || *degrees_end != '\0' || requested_degrees == 0.0) {
            printf("Invalid stepper angle: %s\n", degrees_text);
            return;
        }

        stepper_motor_t *motor = &steppers[motor_index];
        uint32_t requested_frequency_hz = motor->motion.step_frequency_hz;
        if (frequency_text != NULL) {
            char *frequency_end = NULL;
            const unsigned long parsed_frequency = strtoul(frequency_text, &frequency_end, 10);
            if (frequency_end == frequency_text || *frequency_end != '\0' || parsed_frequency == 0u) {
                printf("Invalid step frequency: %s\n", frequency_text);
                return;
            }

            requested_frequency_hz = (uint32_t)parsed_frequency;
        }

        if (!stepper_start_move(requested_degrees, requested_frequency_hz, motor)) {
            printf("Unable to start stepper move. Check angle and frequency.\n");
            return;
        }

        const double magnitude_degrees =
            requested_degrees >= 0.0 ? requested_degrees : -requested_degrees;
        const uint32_t requested_steps =
            (uint32_t)((magnitude_degrees / STEPPER_STEP_ANGLE_DEG) + 0.5);
        const double actual_degrees = (double)requested_steps * STEPPER_STEP_ANGLE_DEG;

        printf("Stepper motor %u move started: %s %lu steps (%.1f deg actual) at %lu Hz\n",
               motor_index + 1u,
               requested_degrees >= 0.0 ? "forward" : "reverse",
               (unsigned long)requested_steps,
               actual_degrees,
               (unsigned long)motor->motion.pwm_config.achieved_frequency_hz);
        return;
    }

    if (strcmp(command, "stop") == 0) {
        char *motor_text = next_token(&context);
        uint motor_index = 0u;

        if (motor_text != NULL) {
            if (!parse_stepper_index(motor_text, &motor_index) || next_token(&context) != NULL) {
                printf("Invalid stepper motor: %s\n", motor_text);
                return;
            }
        }

        steppers[motor_index].motion.move_active = false;
        steppers[motor_index].motion.target_steps = steppers[motor_index].motion.completed_steps;
        stepper_force_idle_output(&steppers[motor_index]);
        printf("Stepper motor %u stopped\n", motor_index + 1u);
        return;
    }

    printf("Unknown command: %s\n", command);
    print_help();
}

int main() {
    stdio_init_all();

    sleep_ms(2000);

    uint16_t dshot_throttle_values[DSHOT_MOTOR_COUNT] = {
        DSHOT_DEFAULT_THROTTLE,
        DSHOT_DEFAULT_THROTTLE,
    };

    if (pwm_generator_calculate(clock_get_hz(clk_sys),
                                STEPPER_DEFAULT_FREQUENCY_HZ,
                                STEPPER_DEFAULT_DUTY_CYCLE,
                                &g_stepper_motors[0].motion.pwm_config) != PWM_GENERATOR_OK) {
        while (true) {
            printf("Stepper PWM configuration failed\n");
            sleep_ms(1000);
        }
    }

    stepper_init();
    for (uint motor_index = 0u; motor_index < STEPPER_MOTOR_COUNT; ++motor_index) {
        g_stepper_motors[motor_index].motion.pwm_config = g_stepper_motors[0].motion.pwm_config;
        stepper_force_idle_output(&g_stepper_motors[motor_index]);
    }

    PIO pio = pio0;
    const uint dshot_state_machines[DSHOT_MOTOR_COUNT] = {0u, 1u};
    const uint dshot_offset = pio_add_program(pio, &dshot_program);

    for (uint motor_index = 0u; motor_index < DSHOT_MOTOR_COUNT; ++motor_index) {
        dshot_init(pio,
                   dshot_state_machines[motor_index],
                   g_dshot_gpios[motor_index],
                   dshot_offset,
                   DSHOT_BITRATE);
    }
    print_help();
    print_status(g_stepper_motors, dshot_throttle_values);

    char serial_line[SERIAL_LINE_BUFFER_SIZE];
    size_t serial_line_length = 0u;
    absolute_time_t next_dshot_send_time = get_absolute_time();

    while (true) {
        const int character = getchar_timeout_us(0);
        if (character != PICO_ERROR_TIMEOUT) {
            if (character == '\r' || character == '\n') {
                if (serial_line_length > 0u) {
                    serial_line[serial_line_length] = '\0';
                    process_serial_command(serial_line, g_stepper_motors, dshot_throttle_values);
                    serial_line_length = 0u;
                }
            } else if (serial_line_length < (SERIAL_LINE_BUFFER_SIZE - 1u)) {
                serial_line[serial_line_length++] = (char)character;
            } else {
                serial_line_length = 0u;
                printf("Input too long\n");
            }
        }

        if (absolute_time_diff_us(get_absolute_time(), next_dshot_send_time) <= 0) {
            for (uint motor_index = 0u; motor_index < DSHOT_MOTOR_COUNT; ++motor_index) {
                dshot_send(pio,
                           dshot_state_machines[motor_index],
                           dshot_throttle_values[motor_index]);
            }
            next_dshot_send_time = delayed_by_us(next_dshot_send_time, DSHOT_SEND_INTERVAL_US);
        }

        tight_loop_contents();
    }
}
