#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Tests a binary with gem5 Full-System (FS) mode by parsing the log output.
# Counterpart to test_with_gem5.sh (SE/syscall-emulation mode): this one is
# for bare-metal + semihosting binaries (no host OS underneath at all — the
# binary boots directly at its ELF entry point via its own crt0, e.g.
# riscv64_baremetal/start_semi.S), so there's no "cpu model" or "arch"
# argument to pass through — the board config is fixed (single MinorCPU,
# see sim_config/gem5_riscv_baremetal_fs.py). ${1}/${2} are accepted purely
# for interface parity with test_with_gem5.sh/test_with_qemu.sh, so
# TEST_SCRIPT definitions in target *.inc files stay visually consistent.
#
# Parameters:
#  ${1} unused (kept for interface parity)
#  ${2} unused (kept for interface parity; no CPU model choice in FS mode)
#  ${3} cross-compiled bare-metal binary to be emulated
#  ${4} - String that is checked for pass/fail.
#  ${5} - target (riscv64_baremetal etc.)
#
# Env vars:
#  GEM5_BIN        - gem5 binary to invoke (default: gem5.opt, must be on PATH)
#  GEM5_FS_CONFIG  - path to the FS-mode board config (default: sim_config/
#                    one directory above this repo checkout, i.e. a sibling
#                    of this tflite-micro clone/submodule — kept out of this
#                    repo since it's a gem5-environment config, not TFLM
#                    source. See the parent repo's README for layout.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_ROOT_DIR=${SCRIPT_DIR}/../../../../

GEM5_BIN="${GEM5_BIN:-gem5.opt}"
GEM5_FS_CONFIG="${GEM5_FS_CONFIG:-${TFLM_ROOT_DIR}../sim_config/gem5_riscv_baremetal_fs.py}"

OUTPUT_DIR="${TFLM_ROOT_DIR}../test/output"
MICRO_LOG_PATH=${OUTPUT_DIR}/${3}
MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs_gem5.txt
M5OUT_DIR=${MICRO_LOG_PATH}/m5out

mkdir -p ${MICRO_LOG_PATH}
rm -rf ${M5OUT_DIR}

${GEM5_BIN} -d ${M5OUT_DIR} ${GEM5_FS_CONFIG} ${3} 2>&1 | tee ${MICRO_LOG_FILENAME}
if [[ ${4} != "non_test_binary" ]]
then
  if grep -q "${4}" ${MICRO_LOG_FILENAME}
  then
    echo "Pass"
    exit 0
  else
    echo "Fail"
    exit 1
  fi
fi
