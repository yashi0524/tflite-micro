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
# Tests a binary with whisper (a functional, non-timing RISC-V ISS) by
# parsing the log output. Alternate TEST_SCRIPT for riscv64_baremetal,
# alongside test_with_gem5_fs.sh — whisper's --semihosting flag services
# the same SYS_WRITE0/SYS_EXIT calls start_semi.S makes, so it can run the
# exact same bare-metal binaries gem5 FS mode does, just as a fast
# functional simulator with no cycle-accurate timing model (see
# sim_config/whisper_rv64gcv_config.json's sibling gem5 config for a
# side-by-side comparison of what each simulator actually models).
#
# Parameters:
#  ${1} unused (kept for interface parity with test_with_gem5_fs.sh)
#  ${2} unused (kept for interface parity; no CPU model choice)
#  ${3} cross-compiled bare-metal binary to be emulated
#  ${4} - String that is checked for pass/fail.
#  ${5} - target (riscv64_baremetal etc.)
#
# Env vars:
#  WHISPER_BIN     - whisper binary to invoke (default: whisper, must be on
#                     PATH)
#  WHISPER_CONFIG  - path to the whisper config JSON (default: sim_config/
#                     one directory above this repo checkout, i.e. a sibling
#                     of this tflite-micro clone/submodule — kept out of
#                     this repo since it's a whisper-environment config, not
#                     TFLM source. See the parent repo's README for layout.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_ROOT_DIR=${SCRIPT_DIR}/../../../../

WHISPER_BIN="${WHISPER_BIN:-whisper}"
WHISPER_CONFIG="${WHISPER_CONFIG:-${TFLM_ROOT_DIR}../sim_config/whisper_rv64gcv_config.json}"

OUTPUT_DIR="${TFLM_ROOT_DIR}../test/output"
MICRO_LOG_PATH=${OUTPUT_DIR}/${3}
MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs_whisper.txt

mkdir -p ${MICRO_LOG_PATH}

${WHISPER_BIN} --configfile ${WHISPER_CONFIG} --semihosting --counters \
  --target ${3} 2>&1 | tee ${MICRO_LOG_FILENAME}
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
