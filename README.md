# CIFAR10 on GAP

## Training

In the cifar10_training.ipynb notebook you can find the training script. I used the google colab with GPU enabled, it can be also run on your local machine.

It produced 2 tflite files:

	- cifar10_model_fp32.tflite
	- cifar10_model_uint8.tflite

## Generate nntool project

With the quantized one you can generate your project. First source the sdk and open an nntool interactive shell environment:

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

For clarity reason I reorganized the folder moving the model and the nntool_script into the *model* folder, changing accordingly the paths in the *common.mk* file. I also added the hwc version for completeness, it can be chosen adding **MODEL_HWC=1** in the make command.

From the training scripts you can save several sample data, I made a directory for them: *samples*

I've then changed the application code to run on real data (images from testing dataset) adding the snippet below:

```
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

This loads data from an image defined by the *AT_IMAGE* in the *Makefile*. It also already shift the [0:255] data into the [-128:127] required by the so trained model.

## Test on your dataset the deployed graph

**Nntool** is a python program which can run inference on data giving bitexact results wrt the real platform. You can import nntool in your python script and use it similarly to the tflite interpreter. *model/test_accuracy.py* shows how to do it.
