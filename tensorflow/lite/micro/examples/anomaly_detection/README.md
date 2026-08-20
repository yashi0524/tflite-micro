# Anomaly detection example (MLPerf Tiny `ad01`)

This is MLPerf Tiny's anomaly-detection reference model
(`ad01_int8.tflite`, from
[mlcommons/tiny](https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection)),
a `640`-in/`640`-out fully int8-quantized dense autoencoder trained on the
DCASE2020 Task 2 ToyCar dataset (predictive-maintenance sound anomaly
detection). Architecture is 10 back-to-back `FULLY_CONNECTED` layers —
`640 -> 128x4 -> 8 (bottleneck) -> 128x4 -> 640` — no convolution, no
LSTM, confirmed by walking the model's flatbuffer operator list directly.

**This test is a build/link/`Invoke()` smoke test, not an accuracy or
quality evaluation** — unlike `../dtln`'s `feature_data`, which is a real
recorded audio spectrogram, `feature_data` here (see
`anomaly_detection_inout_data.cc`) is a fixed-seed synthetic int8 stream,
*not* a real DCASE2020/ToyADMOS log-mel-spectrogram sample. Reproducing a
real 640-wide input (5 frames x 128 mel bins) requires the ToyCar dataset
and mlcommons/tiny's own feature-extraction pipeline
(`benchmark/training/anomaly_detection/{get_dataset.sh,common.py}`),
which this example deliberately doesn't pull in. There is consequently no
golden-reference-output check here, only `Invoke() == kTfLiteOk` and
output tensor shape/type checks.

## Run the tests on a development machine

```
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
make -f tensorflow/lite/micro/tools/make/Makefile test_anomaly_detection_test
```

You should see `~~~ALL TESTS PASSED~~~`. For this project's RISC-V/gem5
setup specifically (source `../../../../../../../2_pattern/tflm/script/0_env_var_setup.sh`
first):

```
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=riscv64_baremetal[_vector] TARGET_TOOLCHAIN_ROOT=$TOOLCHAIN/bin/ \
  TARGET_TOOLCHAIN_PREFIX=riscv-none-elf- BUILD_TYPE=default \
  test_anomaly_detection_test
```

Both `riscv64_baremetal` and `riscv64_baremetal_vector` build and pass
under gem5 (`MinorCPU`) as of this writing — the vectorized target
inherits RVV `Int8DotProductRvv` handling from the same `fully_connected.h`
fast path this project's `dtln` work already vectorized, since this
model uses no op types outside `FULLY_CONNECTED`.
