#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_pi"
UF2_PATH="${BUILD_DIR}/turret.uf2"
GENERATOR="Ninja"

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "Missing toolchain: arm-none-eabi-gcc"
  echo "Install on Debian/Ubuntu/Raspberry Pi OS:"
  echo "  sudo apt update && sudo apt install -y gcc-arm-none-eabi"
  exit 3
fi

# Build firmware
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  OLD_GEN="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1 || true)"
  if [[ -n "${OLD_GEN}" && "${OLD_GEN}" != "${GENERATOR}" ]]; then
    echo "Build dir generator mismatch (${OLD_GEN} != ${GENERATOR}); cleaning ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
  fi
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G "${GENERATOR}"
cmake --build "${BUILD_DIR}" -j

if [[ ! -f "${UF2_PATH}" ]]; then
  echo "UF2 not found at ${UF2_PATH}"
  exit 1
fi

# Resolve mount point (or use first arg)
MOUNT_POINT="${1:-}"
if [[ -z "${MOUNT_POINT}" ]]; then
  for p in \
    "/media/${USER}/RPI-RP2" \
    "/run/media/${USER}/RPI-RP2" \
    "/mnt/RPI-RP2"
  do
    if [[ -d "${p}" ]]; then
      MOUNT_POINT="${p}"
      break
    fi
  done
fi

if [[ -z "${MOUNT_POINT}" || ! -d "${MOUNT_POINT}" ]]; then
  echo "Build complete: ${UF2_PATH}"
  echo "Pico not mounted as RPI-RP2."
  echo "Put Pico in BOOTSEL mode (hold BOOTSEL while plugging in), then run:"
  echo "  ${BASH_SOURCE[0]} /path/to/RPI-RP2"
  exit 2
fi

cp "${UF2_PATH}" "${MOUNT_POINT}/"
sync

echo "Flashed ${UF2_PATH} -> ${MOUNT_POINT}"
echo "Pico will reboot automatically."
