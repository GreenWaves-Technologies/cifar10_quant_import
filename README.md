# CIFAR10 on GAP

## Requirements:
- Install the GAP sdk (>=v4.7.0) from [this link](https://github.com/GreenWaves-Technologies/gap_sdk)

## Training - Testing

### Tensorflow + TFLite

Requirements:
- Tensorflow >=2.7.0

In the *cifar10_training.ipynb* notebook you can find the training script, the model is inspired by the open source [Kaggle Project](https://www.kaggle.com/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy#A-Simple-Keras-CNN-trained-on-CIFAR-10-dataset-with-over-88%-accuracy-(Without-Data-Augmentation)). The model is then converted to tflite w/ and w/o post-training quantization.

- ```model/cifar10_model_fp32.tflite```: Full precision
- ```model/cifar10_model_uint8.tflite```: Quantized

### Pytorch + ONNX

Requirements:
- Pytorch >= 1.12
- onnx >= 1.12.0
- onnxruntime >= 1.11.1

In the *cifar10_training_pytorch.ipynb* notebook you can find the training script, the model architecture is the same as before with few little details to be able to run pytorch Post training quantization (i.e. ReLU6 -> ReLU). The model is trained with pytorch and quantized with [pytorch static post training quantization](https://pytorch.org/docs/stable/quantization.html#). The model is then exported to onnx. 

- ```model/quant_cifar10.onnx```: Quantized

### Model Testing

In each notebook all the models accuracies are tested. In the end the model is opened in NNTool and the accuracy is tested again. Here you can find the results (in parenthesis: SW Backend - Quantization Ranges used - Target (in NNTool different targets might lead to different quantization schemes)):

- ```model/cifar10_model_fp32.tflite``` (TFLite - None - Floating Point): 78.73%
- ```model/cifar10_model_uint8.tflite``` (TFLite - TFlite PTQ - Fixed Point): 77.73%
- ```model/cifar10_model_uint8.tflite``` (NNTool - TFlite PTQ - NE16): 78.0%
- ```model/cifar10_model_uint8.tflite``` (NNTool - TFlite PTQ - SW): 78.4%
- ```model/cifar10_model_fp32.tflite``` (NNTool - NNTool PTQ - NE16): 78.0%
- ```model/cifar10_model_fp32.tflite``` (NNTool - NNTool PTQ - SW): 78.3%
- ```model/quant_cifar10.onnx``` (Pytorch - Pytorch PTQ - Floating Point): 77.59%
- ```model/quant_cifar10.onnx``` (OnnxRuntime - Pytorch PTQ - Floating Point): 77.56%
- ```model/quant_cifar10.onnx``` (NNTool - Pytorch PTQ - NE16): 77.27%
- ```model/quant_cifar10.onnx``` (NNTool - Pytorch PTQ - SW): 77.4%
- ```model/quant_cifar10.onnx``` (NNTool - NNTool PTQ - NE16): 77.53%
- ```model/quant_cifar10.onnx``` (NNTool - NNTool PTQ - SW): 77.4%

## Run Project

To run the porject:

```
source gap_sdk/sourceme.sh
make all run platform=gvsoc \
    MODEL_NE16=1 (use ne16 or SW, default: 1) \
    MODEL_HWC=0 (use HWC tensor layout if SW backend, default: 0) \
    ONNX=0 (use ONNX model, otherwise tflite quantized, default: 0)

```


## Generate nntool project

With the model from onnx/tflite you can directly generate your project. First source the sdk and open an nntool interactive shell environment:

```
source gap_sdk/sourceme.sh
nntool
```

Run the following commands in the nntool shell to prepare the model and generate an already functioning project:

```
open cifar10_model_uint8.tflite -q
adjust
fusions --scale8
gen_project .
```

The default project run on uninitialized input data and the application code is in **cifar10_model_uint8.c**:

```
make clean all run platform=gvsoc
```

## Add application specific code

For the sake of clarity, the project folder has been reorganized moving the tflite model and the nntool_script into the *model* folder, changing accordingly the paths in the *common.mk* file. An HWC tensor layout version of the script has been added as well, it can be chosen adding **MODEL_HWC=1** in the make command.

From the training scripts you can save several sample data, you can find some of them in the *samples* folder.

The application code has then been changed to run on real data (images from testing dataset) adding the snippet below:

```C
#include "gaplib/ImgIO.h"
#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s    ImageName = __XSTR(AT_IMAGE);
...
    #if defined(MODEL_HWC)
    int Traspose2CHW = 0;
    #else
    int Traspose2CHW = 1;
    #endif
    printf("Reading image in %s\n", Traspose2CHW?"CHW":"HWC");
    if (ReadImageFromFile(ImageName, 32, 32, 3, Input_1, 32*32*3*sizeof(char), IMGIO_OUTPUT_CHAR, Traspose2CHW)) {
        printf("Failed to load image %s\n", ImageName);
        pmsis_exit(-1);
    }
    for (int i=0; i<32*32*3; i++) Input_1[i] -= 128;
...
```

This loads data from an image defined by the *AT_IMAGE* in the *Makefile*. It also already shift the \[0:255\] data into the \[-128:127\] required by the so trained model.
NOTE: The data provided in the C code of the application need to be the quantized version of the data expected by the model, from qshow 0 in nntool you can look at the expected range. For example if the input is expected in the \[-1.0:1.0\] real domain range and its data type is int8 \[-128:127\], then -128 will represent -1.0 and 127->0.99.

## Test on your dataset the deployed graph

**Nntool** is a python program which can run inference on data giving bitexact results wrt the real platform. You can import nntool in your python script and use it similarly to the tflite interpreter. *model/test_accuracy.py* shows how to do it.
