cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED ENV{PICO_SDK_PATH})
    set(PICO_SDK_FETCH_FROM_GIT on)
    set(PICO_SDK_FETCH_FROM_GIT_TAG 2.1.1)
endif()

include(pico_sdk_import.cmake)

project(turret C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(turret
    main.c
)

pico_generate_pio_header(turret ${CMAKE_CURRENT_LIST_DIR}/dshot.pio)

target_link_libraries(turret
    pico_stdlib
    hardware_pio
    hardware_pwm
    hardware_gpio
    hardware_clocks
    hardware_irq
)

# Communication over USB (Pi connects via /dev/ttyACM0)
pico_enable_stdio_usb(turret 1)
pico_enable_stdio_uart(turret 0)

pico_add_extra_outputs(turret)
