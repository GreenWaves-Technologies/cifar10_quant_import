NNTOOL=nntool
MODEL_SQ8=1
# MODEL_POW2=1
# MODEL_FP16=1
# MODEL_NE16=1

MODEL_SUFFIX?=
MODEL_PREFIX?=cifar10_model_uint8
MODEL_PYTHON=python3
MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

TRAINED_MODEL = model/cifar10_model_uint8.tflite

MODEL_EXPRESSIONS = 

NNTOOL_EXTRA_FLAGS += 
MODEL_QUANTIZED=1

# Memory sizes for cluster L1, SoC L2 and Flash
TARGET_L1_SIZE = 64000
TARGET_L2_SIZE = 300000
TARGET_L3_SIZE = 8000000

# Cluster stack size for master core and other cores
CLUSTER_STACK_SIZE=4096
CLUSTER_SLAVE_STACK_SIZE=1024
CLUSTER_NUM_CORES=8

MODEL_HWC ?= 0
ifeq ($(MODEL_HWC), 1)
	NNTOOL_SCRIPT = model/nntool_script_hwc
	APP_CFLAGS += -DMODEL_HWC
else
	NNTOOL_SCRIPT = model/nntool_script_chw
endif


$(info GEN ... $(CNN_GEN))
