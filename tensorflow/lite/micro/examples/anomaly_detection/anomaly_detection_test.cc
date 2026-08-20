/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_inout_data.h"
#include "tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/schema/schema_generated.h"

// MLPerf Tiny's `ad01` anomaly-detection benchmark: a 640-in/640-out int8
// dense autoencoder (640->128x4->8->128x4->640, all FULLY_CONNECTED, no
// conv/LSTM -- see ../../../../../../../2_pattern/tflm's
// doc/performance_dtln.md for how this differs from the LSTM-heavy dtln
// model this project's roofline tooling was originally built around).
// Model: https://github.com/mlcommons/tiny, benchmark/training/
// anomaly_detection/trained_models/ad01_int8.tflite.
TEST(AnomalyDetectionTest, TestInvoke) {
  MicroPrintf(
      "\nThis smoke-tests build/link/Invoke() for MLPerf Tiny's ad01 "
      "anomaly-detection autoencoder with a synthetic input -- NOT an "
      "accuracy/quality evaluation. See anomaly_detection_inout_data.cc.\n");

  const tflite::Model* model =
      ::tflite::GetModel(g_anomaly_detection_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // The model is 10 back-to-back FULLY_CONNECTED layers -- no other op
  // types (confirmed by walking the flatbuffer's operator list directly).
  tflite::MicroMutableOpResolver<1> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();

  // Largest activation tensor is the 640-wide input/output; weights are
  // read from the model buffer, not the arena. 8 KiB is generous headroom
  // over dtln's 16 KiB (which also carries two LSTM layers' state).
  constexpr int tensor_arena_size = 8 * 1024;
  alignas(16) uint8_t tensor_arena[tensor_arena_size];

  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);
  EXPECT_NE(input, nullptr);
  EXPECT_EQ(2, input->dims->size);
  EXPECT_EQ(1, input->dims->data[0]);
  EXPECT_EQ(640, input->dims->data[1]);
  EXPECT_EQ(kTfLiteInt8, input->type);

  for (size_t i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = feature_data[i];
  }

  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }
  EXPECT_EQ(kTfLiteOk, invoke_status);

  TfLiteTensor* output = interpreter.output(0);
  EXPECT_EQ(2, output->dims->size);
  EXPECT_EQ(1, output->dims->data[0]);
  EXPECT_EQ(640, output->dims->data[1]);
  EXPECT_EQ(kTfLiteInt8, output->type);

  MicroPrintf("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_MAIN
