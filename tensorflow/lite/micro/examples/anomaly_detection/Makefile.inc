
ANOMALY_DETECTION_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_inout_data.cc

ANOMALY_DETECTION_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_inout_data.h

ANOMALY_DETECTION_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_int8.tflite

ANOMALY_DETECTION_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_int8_model_data.cc

ANOMALY_DETECTION_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/anomaly_detection_int8_model_data.h

#Find any platform - specific rules for this example.
include $(wildcard $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/anomaly_detection/*/Makefile.inc)

# Tests loading and running MLPerf Tiny's ad01 anomaly-detection model.
$(eval $(call microlite_test,anomaly_detection_test,\
$(ANOMALY_DETECTION_TEST_SRCS),$(ANOMALY_DETECTION_TEST_HDRS),$(ANOMALY_DETECTION_GENERATOR_INPUTS)))
