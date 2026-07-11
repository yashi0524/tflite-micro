# Vector-enabled sibling of riscv64_baremetal_makefile.inc — identical in
# every respect (same bare-metal M-mode + semihosting crt0/linker script,
# same gem5 FS / whisper simulators) except RISCV_ARCH adds the Zve64x
# extension (integer-only embedded vector profile, ELEN=64 — matches
# sim_config/gem5_riscv_baremetal_fs.py's RiscvISA(vlen=512, elen=64) and
# sim_config/whisper_rv64gcv_config.json's vector config; no changes needed
# to either sim config, they were already vector-capable).
#
# Kept as a genuinely separate TARGET (not just a RISCV_ARCH override on
# riscv64_baremetal) deliberately: TFLM's GENDIR path
# (gen/<TARGET>_<host>_<build_type>_<toolchain>/) is keyed on TARGET name,
# not on RISCV_ARCH, so overriding RISCV_ARCH on the same TARGET risks
# silently reusing stale .o files compiled with the old -march= — a real
# correctness hazard, not just a style preference. Two TARGETs means two
# GENDIRs, no risk of cross-contamination.
#
# Why this exists at all: tensorflow/lite/kernels/internal/reference/
# integer_ops/fully_connected.h has an `#if defined(__riscv_vector)`
# RVV-intrinsics fast path for the int8-quantized FullyConnected loop
# (Int8DotProductRvv) — inert everywhere else (including plain
# riscv64_baremetal, which doesn't define __riscv_vector), active only when
# compiled with a vector-capable -march= like this target's. Built to
# compare cycle counts against the scalar baseline for
# dtln_noise_suppression.tflite's FULLY_CONNECTED layer (M=1, K=128, N=257).
#
# Usage:
#   make -f tensorflow/lite/micro/tools/make/Makefile \
#     TARGET=riscv64_baremetal_vector \
#     TARGET_TOOLCHAIN_ROOT=<aarch64-native toolchain>/bin/ \
#     TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
#     test_hello_world_test

TARGET_ARCH := riscv64
TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

# Zve64x: integer-only embedded vector extension, ELEN=64 -- no F/D needed
# for this int8-only workload, unlike gemm's rv64gcv (which needs full V
# since it does FP32/FP64 GEMM). zicsr still required for start_semi.S's
# CSR instructions, same as plain riscv64_baremetal.
RISCV_ARCH := rv64imc_zicsr_zve64x
RISCV_ABI := lp64
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

LDFLAGS += \
  -nostartfiles \
  -nostdlib \
  -T $(TENSORFLOW_ROOT)tensorflow/lite/micro/riscv64_baremetal_vector/linker_semi.ld

MICROLITE_LIBS := -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

MICROLITE_CC_SRCS += $(TENSORFLOW_ROOT)tensorflow/lite/micro/riscv64_baremetal_vector/start_semi.S

# See http://b/158651472 for why memory arena threshold test is disabled.
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

CCFLAGS += $(RISCV_EXTRA_CFLAGS)
CXXFLAGS += $(RISCV_EXTRA_CFLAGS)

LDFLAGS += -mno-relax

SIMULATOR ?= gem5
ifeq ($(SIMULATOR),whisper)
  TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_whisper.sh riscv64 minor
else
  TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_gem5_fs.sh riscv64 minor
endif
SIZE_SCRIPT := ${TENSORFLOW_ROOT}tensorflow/lite/micro/testing/size_riscv32_binary.sh

include $(MAKEFILE_DIR)/ext_libs/eyalroz_printf.inc
