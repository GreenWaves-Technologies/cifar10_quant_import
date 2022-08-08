import argparse
import argcomplete
from pathlib import Path
from nntool.api import NNGraph
import numpy as np
from tqdm import tqdm

def create_parser():
	# create the top-level parser
	parser = argparse.ArgumentParser(prog='test_gapflow')

	parser.add_argument('--model_path', type=str, required=True,
						help='path to model to test')
	parser.add_argument('--n_samples', type=int, default=10000,
						help='number of samples to test on, default: all')
	parser.add_argument('--quantize_in_nntool', action="store_true",
						help='quantize in nntool, otherwise use loaded quantization ranges')
	parser.add_argument('--ne16', action="store_true",
						help='use ne16 quantization scheme')
	return parser

def main():
	parser = create_parser()
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	n_samples = args.n_samples if args.n_samples else 10000
	model_path = Path(args.model_path)
	print(model_path)
	if model_path.suffix == ".tflite":
		from tensorflow.keras.datasets import cifar10
		from tensorflow.keras.utils import to_categorical
		import tensorflow as tf
		def nntool_inference(graph, test_images, test_labels):
			predictions = np.zeros((len(test_images),), dtype=int)
			for i, (test_image, test_label) in enumerate(tqdm(zip(test_images, test_labels), total=len(test_labels))):
				output = graph.execute([test_image], dequantize=True)
				predictions[i] = output[-1][0].argmax()

			test_labels_not_one_hot = np.argmax(test_labels, 1)
			accuracy = (np.sum(test_labels_not_one_hot == predictions) * 100) / len(test_images)
			return accuracy

		(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

		# Converting the pixels data to float type
		train_images = train_images.astype('float32')
		test_images = test_images.astype('float32')
		 
		# Standardizing (255 is the total number of pixels an image can have)
		train_images = (train_images / 128) - 1.0
		test_images = (test_images / 128) - 1.0

		# One hot encoding the target class (labels)
		num_classes = 10
		train_labels = to_categorical(train_labels, num_classes)
		test_labels = to_categorical(test_labels, num_classes)

		G = NNGraph.load_graph(str(model_path), load_quantization=True)
		if args.quantize_in_nntool:
			stats = G.collect_statistics([np.array(data) for data in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100)])
		else:
			stats = None

		G.quantize(stats, graph_options={"use_ne16": args.ne16, "hwc": True})
		G.adjust_order()
		G.fusions("scaled_match_group")
		nntool_accuracy = nntool_inference(G, test_images[:n_samples], test_labels[:n_samples])

	elif model_path.suffix == ".onnx":
		import torch
		import torchvision
		import torchvision.transforms as transforms

		def nntool_inference(graph, dataloader, max_iter=10000):
			total = 0
			correct = 0
			for data in tqdm(dataloader):
				images, labels = data[0].to("cpu").detach().numpy(), data[1].to("cpu").detach().numpy()

				outputs = []
				for image, label in tqdm(zip(images, labels), total=len(images)):
					# calculate outputs by running images through the network
					input_data = image.transpose(1,2,0)
					output = G.execute([input_data], dequantize=True)[-1][0]
					outputs.append(output)

				# the class with the highest energy is what we choose as prediction
				predicted = np.argmax(np.array(outputs), -1)
				total += labels.size
				correct += (predicted == labels).sum().item()
				if total >= max_iter:
					return correct / total

			return correct / total

		home_dir = Path.home()
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])
		batch_size = 128
		testset = torchvision.datasets.CIFAR10(root=Path.joinpath(home_dir, 'Datasets/cifar10'), train=False,
											   download=True, transform=transform)
		testset = torch.utils.data.Subset(testset, list(range(0, n_samples)))
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
												 shuffle=False, num_workers=2)
		G = NNGraph.load_graph(str(model_path))
		if args.quantize_in_nntool:
			trainset = torchvision.datasets.CIFAR10(root=Path.joinpath(home_dir, 'Datasets/cifar10'), train=True,
													download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
													  shuffle=True, num_workers=2)
			calibration_input, calibration_target = next(iter(trainloader))
			stats = G.collect_statistics([data.to("cpu").detach().numpy() for data in calibration_input])
		else:
			stats = None

		G.quantize(None, graph_options={"use_ne16": args.ne16, "hwc": True})
		G.adjust_order()
		G.fusions("scaled_match_group")
		nntool_accuracy = nntool_inference(G, testloader, n_samples)

	print(f"NNTool Accuracy: {nntool_accuracy * 100:.2f}%")

if __name__ == "__main__":
	main()
