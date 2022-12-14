NNTOOL=nntool
MODEL_SQ8=1

MODEL_SUFFIX?=
MODEL_PREFIX?=cifar10_model
MODEL_PYTHON=python3
MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

ifeq ($(ONNX), 1)
TRAINED_MODEL = model/quant_cifar10.onnx
else
TRAINED_MODEL = model/cifar10_model_uint8.tflite
MODEL_QUANTIZED=1
APP_CFLAGS = -DOUTPUT_SHORT
endif

MODEL_EXPRESSIONS = 

NNTOOL_EXTRA_FLAGS += 

# Memory sizes for cluster L1, SoC L2 and Flash
ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
	FREQ_CL?=375
	FREQ_FC?=375
	TARGET_L1_SIZE = 128000
	TARGET_L2_SIZE = 1300000
	TARGET_L3_SIZE = 8000000
else
	FREQ_CL?=175
	FREQ_FC?=250
	TARGET_L1_SIZE = 64000
	TARGET_L2_SIZE = 300000
	TARGET_L3_SIZE = 8000000
endif


# Cluster stack size for master core and other cores
CLUSTER_STACK_SIZE=2048
CLUSTER_SLAVE_STACK_SIZE=512
CLUSTER_NUM_CORES=8

MODEL_HWC ?= 0
MODEL_NE16?= 1
ifneq '$(TARGET_CHIP_FAMILY)' 'GAP9'
	MODEL_NE16=0
endif

NNTOOL_SCRIPT = model/nntool_script_chw
ifeq ($(MODEL_HWC), 1)
	NNTOOL_SCRIPT = model/nntool_script_hwc
	APP_CFLAGS += -DMODEL_HWC
endif
ifeq ($(MODEL_NE16), 1)
	NNTOOL_SCRIPT = model/nntool_script_ne16
	APP_CFLAGS += -DMODEL_NE16
endif


$(info GEN ... $(CNN_GEN))
