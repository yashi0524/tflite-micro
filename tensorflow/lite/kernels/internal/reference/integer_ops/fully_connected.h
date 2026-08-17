/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/common.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif  // defined(__riscv_vector)

namespace tflite {
namespace reference_integer_ops {

#if defined(__riscv_vector)
// RVV-vectorized replacement for the innermost `for (d) acc += (filter_val +
// filter_offset) * (input_val + input_offset)` reduction in the int8
// FullyConnected() below.
//
// Uses the standard gemmlowp-style algebraic expansion instead of folding
// filter_offset/input_offset directly into a widening vwadd.vx:
//   Sigma_d (filter[d]+filter_offset)*(input[d]+input_offset) =
//     Sigma(filter*input) + input_offset*Sigma(filter) +
//     filter_offset*Sigma(input) + accum_depth*filter_offset*input_offset
// filter_offset/input_offset are only ever applied via plain int32_t scalar
// arithmetic *after* the vector reduction -- never passed as a vector
// intrinsic's scalar operand. This matters because that operand is
// SEW(=8)-wide for a widening op sourced from int8: an offset of exactly
// 128 (a legal value -- e.g. zero_point=-128 gives input_offset=128, and
// int8's range is -128..127) silently truncates to -128 when passed to
// vwadd.vx, corrupting the result. Confirmed via a standalone probe: the
// previous (int8_t input_offset/filter_offset folded into vwadd.vx)
// implementation produced wrong output on ~1/3 of (accum_depth,
// offset) combinations tested whenever an offset was exactly +/-128;
// this expansion passed all of them. Both widens below use offset=0 (pure
// sign-extend), which can never overflow int8_t regardless of what
// input_offset/filter_offset actually are.
//
// Requires GCC >= 13.4 (xPack riscv-none-elf-gcc): this exact intrinsic
// sequence miscompiles at -O1/-O2 under GCC 13.2.0 -- whisper traps with
// an illegal instruction, gem5 runs to completion but produces wrong
// output. Verified fixed under 13.4.0 (this project's toolchain as of
// this fix) via a standalone probe (114/114 correct at -O2, matching the
// -O0 result) before applying here.
//
// Base LMUL=2 (e8m2 -> e16m4 -> e32m8): the widest feasible given this
// double-widening chain -- a third widening step (base m4) would need
// m16 for the second widen, which doesn't exist (RVV's max LMUL is 8).
// Measured on gem5 (cycle-accurate) across every FC/LSTM-gate accum_depth
// exercised by this project's models: 25-34% fewer cycles than base
// LMUL=1 for accum_depth >= 128 (dtln's FC/LSTM shapes: 128, 257;
// micro_speech's FC: 4000), ~4% *more* cycles for accum_depth <= 28
// (mnist_lstm's LSTM gates: 20, 28; hello_world's FC: 16) -- those already
// complete in a single vector op at LMUL=1 (VLEN=512 gives VLMAX=64 for
// e8), so LMUL=2 only adds fixed per-instruction overhead there with no
// iteration-count reduction to offset it. Net win for this project's
// standing benchmark (dtln, all shapes >=128); small regression accepted
// on the other models' already-cheap small-K gates rather than adding a
// runtime branch to the hot path.
inline int32_t Int8DotProductRvv(const int8_t* input, const int8_t* filter,
                                 int accum_depth, int32_t input_offset,
                                 int32_t filter_offset) {
  int32_t dot = 0;
  int32_t filter_sum = 0;
  int32_t input_sum = 0;
  int n = accum_depth;
  const int8_t* a = input;
  const int8_t* w = filter;
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e8m2(n);
    vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
    vint8m2_t vw = __riscv_vle8_v_i8m2(w, vl);
    vint16m4_t va16 = __riscv_vwadd_vx_i16m4(va, 0, vl);
    vint16m4_t vw16 = __riscv_vwadd_vx_i16m4(vw, 0, vl);
    vint32m8_t prod32 = __riscv_vwmul_vv_i32m8(vw16, va16, vl);
    vint32m1_t zero32 = __riscv_vmv_v_x_i32m1(0, 1);

    vint32m1_t dot_v = __riscv_vredsum_vs_i32m8_i32m1(prod32, zero32, vl);
    dot += __riscv_vmv_x_s_i32m1_i32(dot_v);

    vint32m1_t fsum_v = __riscv_vwredsum_vs_i16m4_i32m1(vw16, zero32, vl);
    filter_sum += __riscv_vmv_x_s_i32m1_i32(fsum_v);

    vint32m1_t isum_v = __riscv_vwredsum_vs_i16m4_i32m1(va16, zero32, vl);
    input_sum += __riscv_vmv_x_s_i32m1_i32(isum_v);

    a += vl;
    w += vl;
    n -= static_cast<int>(vl);
  }
  return dot + input_offset * filter_sum + filter_offset * input_sum +
         accum_depth * filter_offset * input_offset;
}
#endif  // defined(__riscv_vector)

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const InputType* input_data, const RuntimeShape& filter_shape,
    const WeightType* filter_data, const RuntimeShape& bias_shape,
    const BiasType* bias_data, const RuntimeShape& output_shape,
    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();

  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

// This implementation receives the scales in float and performs requant in
// float to avoid loss of precision.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const InputType* input_data, const RuntimeShape& filter_shape,
    const WeightType* filter_data, const RuntimeShape& bias_shape,
    const BiasType* bias_data, const RuntimeShape& output_shape,
    float input_scale, float output_scale, const float* filter_scales,
    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();

  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }

      const float scale = filter_scales[out_c];
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t acc_scaled = static_cast<int32_t>(
          round(static_cast<double>(acc) * effective_output_scale));

      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape,
                    const InputType* input_data,
                    const RuntimeShape& filter_shape,
                    const WeightType* filter_data,
                    const RuntimeShape& bias_shape, const BiasType* bias_data,
                    const RuntimeShape& output_shape, OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
#if defined(__riscv_vector)
      // Only the int8-in/int8-filter/int32-bias instantiation matches the
      // RVV helper's assumptions (see Int8DotProductRvv's comment) -- other
      // instantiations of this template fall through to the portable scalar
      // loop below unchanged. OutputType is deliberately not part of this
      // check: Int8DotProductRvv only ever produces the int32 accumulation,
      // identical to the scalar loop -- OutputType only affects the
      // requantize/clamp/cast below, which is unchanged either way. This
      // lets UNIDIRECTIONAL_SEQUENCE_LSTM's int16-output gate matmuls
      // (lstm_eval.cc's FullyConnected() call, same int8/int8/int32
      // in/filter/bias types as FC) take this path too.
      if constexpr (std::is_same<InputType, int8_t>::value &&
                    std::is_same<WeightType, int8_t>::value &&
                    std::is_same<BiasType, int32_t>::value) {
        acc = Int8DotProductRvv(input_data + b * accum_depth,
                                filter_data + out_c * accum_depth,
                                accum_depth, input_offset, filter_offset);
      } else {
        for (int d = 0; d < accum_depth; ++d) {
          int32_t input_val = input_data[b * accum_depth + d];
          int32_t filter_val = filter_data[out_c * accum_depth + d];
          acc += (filter_val + filter_offset) * (input_val + input_offset);
        }
      }
#else   // !defined(__riscv_vector)
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
#endif  // defined(__riscv_vector)
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

// This implementation receives the scales in float and performs requant in
// float to avoid loss of precision.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape,
                    const InputType* input_data,
                    const RuntimeShape& filter_shape,
                    const WeightType* filter_data,
                    const RuntimeShape& bias_shape, const BiasType* bias_data,
                    const RuntimeShape& output_shape, float input_scale,
                    float output_scale, float filter_scale,
                    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            static_cast<double>(filter_scale) /
                                            static_cast<double>(output_scale);
      int32_t acc_scaled = static_cast<int32_t>(
          round(static_cast<double>(acc) * effective_output_scale));
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
