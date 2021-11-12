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
from utils.data_importer import import_data

dataset_files = glob.glob("samples/*")
G = NNToolShell.get_graph_from_commands(['open BUILD_MODEL/cifar10_model_uint8.json'])
executer = GraphExecuter(G, qrecs=G.quantization)

for file in dataset_files:
	print(file)
	# Transpose HWC to CHW
	input_tensor = import_data(file, norm_func='x:x/128-1', transpose=True)

	outputs = executer.execute(input_tensor, qmode=QuantizationMode.all(), silent=True)
	print(outputs[-1][0])

