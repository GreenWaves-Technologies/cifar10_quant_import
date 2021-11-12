# You need to have sourced the SDK
# beware of utils submodule because there is one in nntool
import sys
import os
sys.path.insert(0, os.environ['NNTOOL_DIR'])
import glob
from PIL import Image
import numpy as np

# NNTOOL Modules
from execution.graph_executer import GraphExecuter
from execution.quantization_mode import QuantizationMode
from interpreter.nntool_shell import NNToolShell

dataset_files = glob.glob("samples/*")
G = NNToolShell.get_graph_from_commands(['open BUILD_MODEL/cifar10_model_uint8.json'])
executer = GraphExecuter(G, qrecs=G.quantization)

for file in dataset_files:
	print(file)
	img = Image.open(file)
	# Transpose HWC to CHW
	input_tensor = np.array(img).transpose((2, 0, 1))
	# Even in quantized mode nntool requires float data exactly as you trained the model -> [-1.0: 1.0]
	input_tensor = input_tensor.astype(np.float32) / 128 - 1

	outputs = executer.execute([input_tensor], qmode=QuantizationMode.all(), silent=True)
	pred_class = np.argmax(outputs[-1][0])
	confidence = outputs[-1][0][pred_class]
	print(f"Predicted Class: {pred_class} with confidence: {confidence}")

