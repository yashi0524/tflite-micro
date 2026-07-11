# Settings for RISCV 32-bit toolchain.
TARGET_ARCH := riscv32
TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

RISCV_ARCH := rv32imc
RISCV_ABI := ilp32
RISCV_CODE_MODEL := medany

# Allow additional flags on the command line for debugging.
RISCV_EXTRA_CFLAGS :=

TARGET_DEFAULT_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/riscv_toolchain/bin/
TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
$(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))

export PATH := $(TARGET_TOOLCHAIN_ROOT):$(PATH)

PLATFORM_FLAGS = \
  -march=$(RISCV_ARCH) \
  -mabi=$(RISCV_ABI) \
  -mcmodel=$(RISCV_CODE_MODEL) \
  -mexplicit-relocs \
  -fno-builtin-printf \
  -DTF_LITE_MCU_DEBUG_LOG \
  -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
  -funsigned-char \
  -fno-delete-null-pointer-checks \
  -fomit-frame-pointer

CXXFLAGS += $(PLATFORM_FLAGS) \
  -fpermissive \
  -fno-use-cxa-atexit \
  -DTF_LITE_USE_GLOBAL_MIN \
  -DTF_LITE_USE_GLOBAL_MAX

CCFLAGS += $(PLATFORM_FLAGS)

BUILD_TYPE := micro

LDFLAGS += --specs=nano.specs

# See http://b/158651472 for why memory arena threshold test is disabled.
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

CCFLAGS += $(RISCV_EXTRA_CFLAGS)
CXXFLAGS += $(RISCV_EXTRA_CFLAGS)

# This disables the "linker relaxation" optimization, which produced incorrect code.
# TODO(b/279805615): Check whether this is fixed in newer versions of the toolchain.
LDFLAGS += -mno-relax

# SIMULATOR selects how TEST_SCRIPT runs a built test binary: "qemu" (the
# only supported choice — via QEMU linux-user mode). Override on the
# command line if ever needed, e.g.:
#   make ... TARGET=riscv32_generic SIMULATOR=qemu test_hello_world_test
#
# SIMULATOR=gem5 (gem5 SE/syscall-emulation mode, via
# tensorflow/lite/micro/testing/test_with_gem5.sh) is intentionally
# disabled here: riscv64_baremetal + SIMULATOR=whisper now covers the
# fast-functional-simulator role gem5 SE mode was filling, and gem5 FS
# mode (riscv64_baremetal's default) covers the cycle-accurate role — so
# gem5 SE mode on this target is redundant. test_with_gem5.sh and
# sim_config/gem5_riscv_se.py are left in place (not deleted) in case
# this is ever revisited.
SIMULATOR ?= qemu
ifeq ($(SIMULATOR),gem5)
  $(error SIMULATOR=gem5 (SE mode) is disabled for $(TARGET) — use \
    TARGET=riscv64_baremetal with SIMULATOR=whisper (fast, functional) or \
    the default SIMULATOR=gem5 there (cycle-accurate FS mode) instead)
else
  TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_qemu.sh riscv32 rv32
endif
SIZE_SCRIPT := ${TENSORFLOW_ROOT}tensorflow/lite/micro/testing/size_riscv32_binary.sh

include $(MAKEFILE_DIR)/ext_libs/eyalroz_printf.inc
