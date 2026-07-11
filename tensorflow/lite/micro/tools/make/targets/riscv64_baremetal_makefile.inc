# Settings for a bare-metal RISC-V 64-bit M-mode target, run under gem5
# Full-System (FS) mode with RISC-V semihosting I/O — no host OS/kernel/
# libc-syscall-emulation underneath at all (contrast with riscv64_generic,
# which runs ordinary newlib+Linux-ABI-ecall binaries under QEMU linux-user
# mode or gem5 SE mode). The binary boots directly at its ELF entry point
# via its own crt0 (start_semi.S: stack/gp/mtvec/BSS-zero/.data-copy setup),
# with _write/_exit implemented via semihosting SYS_WRITE0/SYS_EXIT instead
# of Linux syscalls.
#
# There is intentionally no SIMULATOR switch here (unlike riscv{32,64}_generic):
# QEMU linux-user mode cannot run this binary at all (it's not a Linux
# process — no ELF interpreter, no syscalls to emulate), so gem5 FS mode
# (test_with_gem5_fs.sh) is the only supported way to run it.
#
# Usage:
#   make -f tensorflow/lite/micro/tools/make/Makefile \
#     TARGET=riscv64_baremetal \
#     TARGET_TOOLCHAIN_ROOT=<aarch64-native toolchain>/bin/ \
#     TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- \
#     test_hello_world_test

TARGET_ARCH := riscv64
TARGET_TOOLCHAIN_PREFIX := riscv64-unknown-elf-

# zicsr is required for start_semi.S's CSR instructions (mtvec/mstatus/
# mcause/mepc/mtval) — plain rv64imc (as used by riscv64_generic) lacks it.
RISCV_ARCH := rv64imc_zicsr
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

# No --specs=nano.specs here (unlike riscv{32,64}_generic): this target
# links directly against the toolchain's plain libc/libgcc for the
# rv64imc/lp64 multilib via -nostartfiles/-nostdlib + our own crt0/linker
# script below, so nano-newlib's separate spec file doesn't apply.
LDFLAGS += \
  -nostartfiles \
  -nostdlib \
  -T $(TENSORFLOW_ROOT)tensorflow/lite/micro/riscv64_baremetal/linker_semi.ld

# -nostdlib drops the default -lc/-lm/-lgcc auto-linking (it implies
# -nodefaultlibs too, not just -nostartfiles) — add them back explicitly.
# --start-group/--end-group resolves the circular libc<->libgcc symbol
# references (soft-float/int64 helpers newlib's stdio pulls in).
MICROLITE_LIBS := -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

# Our own crt0 — compiled and archived into libtensorflow-microlite.a
# via the standard $(CORE_OBJDIR)%.o: %.S rule, then linked automatically
# since $(MICROLITE_LIB_PATH) is already part of every %_test link line.
MICROLITE_CC_SRCS += $(TENSORFLOW_ROOT)tensorflow/lite/micro/riscv64_baremetal/start_semi.S

# See http://b/158651472 for why memory arena threshold test is disabled.
EXCLUDED_TESTS := \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

CCFLAGS += $(RISCV_EXTRA_CFLAGS)
CXXFLAGS += $(RISCV_EXTRA_CFLAGS)

# This disables the "linker relaxation" optimization, which produced incorrect code.
# TODO(b/279805615): Check whether this is fixed in newer versions of the toolchain.
LDFLAGS += -mno-relax

TEST_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/testing/test_with_gem5_fs.sh riscv64 minor
SIZE_SCRIPT := ${TENSORFLOW_ROOT}tensorflow/lite/micro/testing/size_riscv32_binary.sh

include $(MAKEFILE_DIR)/ext_libs/eyalroz_printf.inc
