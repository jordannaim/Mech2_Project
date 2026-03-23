from __future__ import annotations

from time import time

from .models import RuntimeState, SystemMode


def set_fault(state: RuntimeState, reason: str) -> RuntimeState:
    state.fault = reason
    state.armed = False
    state.fire_in_progress = False
    state.mode = SystemMode.FAULT
    state.updated_at = time()
    return state


def clear_fault(state: RuntimeState) -> RuntimeState:
    state.fault = ""
    if state.yaw_homed:
        state.mode = SystemMode.READY_IDLE
    else:
        state.mode = SystemMode.HOMING_YAW
    state.updated_at = time()
    return state


def mark_boot(state: RuntimeState) -> RuntimeState:
    state.mode = SystemMode.BOOT
    state.updated_at = time()
    return state


def begin_homing(state: RuntimeState) -> RuntimeState:
    state.mode = SystemMode.HOMING_YAW
    state.yaw_homed = False
    state.armed = False
    state.updated_at = time()
    return state


def complete_homing(state: RuntimeState) -> RuntimeState:
    state.yaw_homed = True
    state.mode = SystemMode.READY_IDLE
    state.updated_at = time()
    return state


def set_pitch_confirmed(state: RuntimeState, confirmed: bool) -> RuntimeState:
    state.pitch_zero_assumed = confirmed
    if state.mode == SystemMode.BOOT:
        state.mode = SystemMode.HOMING_YAW if not state.yaw_homed else SystemMode.READY_IDLE
    state.updated_at = time()
    return state


def arm(state: RuntimeState) -> RuntimeState:
    if state.mode == SystemMode.FAULT:
        return state
    if state.yaw_homed and state.pitch_zero_assumed:
        state.armed = True
        state.mode = SystemMode.ARMED
    state.updated_at = time()
    return state


def disarm(state: RuntimeState) -> RuntimeState:
    state.armed = False
    if not state.fault:
        state.mode = SystemMode.READY_IDLE if state.yaw_homed else SystemMode.HOMING_YAW
    state.updated_at = time()
    return state


def mark_firing(state: RuntimeState) -> RuntimeState:
    state.fire_in_progress = True
    state.mode = SystemMode.FIRING
    state.updated_at = time()
    return state


def mark_fire_complete(state: RuntimeState) -> RuntimeState:
    state.fire_in_progress = False
    state.last_fire_ts = time()
    state.fire_count += 1
    state.mode = SystemMode.ARMED if state.armed else SystemMode.READY_IDLE
    state.updated_at = time()
    return state
