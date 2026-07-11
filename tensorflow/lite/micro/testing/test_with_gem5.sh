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
# Tests a binary with gem5 (syscall-emulation mode) by parsing the log
# output. Mirrors test_with_qemu.sh's interface/behavior so it's a drop-in
# TEST_SCRIPT alternative for RISC-V targets.
# Parameters:
#  ${1} unused (kept for interface parity with test_with_qemu.sh)
#  ${2} gem5 CPU model: atomic | timing | minor
#  ${3} cross-compiled binary to be emulated
#  ${4} - String that is checked for pass/fail.
#  ${5} - target (riscv32_generic etc.)
#
# Env vars:
#  GEM5_BIN        - gem5 binary to invoke (default: gem5.opt, must be on PATH)
#  GEM5_SE_CONFIG  - path to the SE-mode board config (default: sim_config/
#                    one directory above this repo checkout, i.e. a sibling
#                    of this tflite-micro clone/submodule — kept out of this
#                    repo since it's a gem5-environment config, not TFLM
#                    source. See the parent repo's README for layout.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_ROOT_DIR=${SCRIPT_DIR}/../../../../

GEM5_BIN="${GEM5_BIN:-gem5.opt}"
GEM5_SE_CONFIG="${GEM5_SE_CONFIG:-${TFLM_ROOT_DIR}../sim_config/gem5_riscv_se.py}"

TEST_TMPDIR=/tmp/test_${5}
MICRO_LOG_PATH=${TEST_TMPDIR}/${3}
MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
M5OUT_DIR=${MICRO_LOG_PATH}/m5out

mkdir -p ${MICRO_LOG_PATH}
rm -rf ${M5OUT_DIR}

${GEM5_BIN} -d ${M5OUT_DIR} ${GEM5_SE_CONFIG} --cpu=${2} ${3} 2>&1 | tee ${MICRO_LOG_FILENAME}
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
