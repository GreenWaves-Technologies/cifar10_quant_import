import glob
from PIL import Image
import numpy as np
from nntool.api import NNGraph

quant_model_path = "model/cifar10_model_uint8.tflite"
G = NNGraph.load_graph(quant_model_path, load_quantization=True)
G.quantize(
    statistics=None,
    graph_options={
        "ne16": False,
        "hwc": False,
    }
)
G.adjust_order()
G.add_dimensions()
G.fusions('scaled_match_group')

dataset_files = glob.glob("samples/*")
for file in dataset_files:
	print(file)
	img = Image.open(file)
	# Transpose HWC to CHW
	input_tensor = np.array(img).transpose((2, 0, 1))
	# Even in quantized mode nntool requires float data exactly as you trained the model -> [-1.0: 1.0]
	input_tensor = input_tensor.astype(np.float32) / 128 - 1

	outputs = G.execute([input_tensor], quantize=True, dequantize=True)
	pred_class = np.argmax(outputs[-1][0])
	confidence = outputs[-1][0][pred_class]
	print(f"Predicted Class: {pred_class} with confidence: {confidence}")
