/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// riscv64_baremetal implementation of the micro_time.h interface, backed by
// the RISC-V `mcycle` CSR (M-mode cycle counter, CSR 0xB00) — read directly
// rather than via the `rdcycle` pseudo-instruction (which targets the
// unprivileged shadow CSR `cycle`, 0xC00). This crt0 runs entirely in
// M-mode, so `mcycle` is always accessible regardless of the target's
// declared ISA extension set; `cycle` is not — whisper traps it as an
// illegal instruction unless the ISA string explicitly includes Zicntr
// (gem5 is more lenient and doesn't enforce this), which
// sim_config/whisper_rv64gcv_config.json doesn't currently declare. This
// matches the sibling `gemm` project's own kernels (src/gemm.c etc.), which
// all read `mcycle` for exactly this reason. Both gem5's RiscvMinorCPU
// (cycle-accurate) and whisper (functional-only, tracked via its own HPM
// counters) correctly emulate this counter, so ticks returned here are real
// simulated cycles, not host wall-clock time.
//
// Without this file, tensorflow/lite/micro/micro_time.cc's reference
// implementation (ticks_per_second()/GetCurrentTimeTicks() both hardcoded to
// return 0) is used instead — which is why every op's profiled duration
// printed "0 ticks (0 ms)" before this file existed.

#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

// Nominal 1 GHz, matching sim_config/gem5_riscv_baremetal_fs.py's
// system.clk_domain.clock = "1GHz" — makes TicksToMs()'s output meaningful
// for gem5 runs. whisper has no clock-domain concept (functional-only), so
// for whisper runs this value only affects the ms conversion, not the raw
// tick/cycle counts themselves (which are what actually matters for
// relative per-op comparison).
uint32_t ticks_per_second() { return 1000000000; }

uint32_t GetCurrentTimeTicks() {
  uint64_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return static_cast<uint32_t>(cycles);
}

}  // namespace tflite
