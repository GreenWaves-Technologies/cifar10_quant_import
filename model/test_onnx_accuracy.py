import argparse
import argcomplete
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort

def create_parser():
	# create the top-level parser
	parser = argparse.ArgumentParser(prog='test_gapflow')

	parser.add_argument('--model_path', type=str, required=True,
						help='path to model to test')
	parser.add_argument('--n_samples', type=int, default=10000,
						help='number of samples to test on, default: all')
	return parser

def onnx_inference(model_path, dataloader):
	sess = ort.InferenceSession(model_path)
	input_name = sess.get_inputs()[0].name
	total = 0
	correct = 0
	for data in tqdm(dataloader):
		images, labels = data[0].to("cpu").detach().numpy(), data[1].to("cpu").detach().numpy()

		outputs = []
		for image, label in zip(images, labels):
			# calculate outputs by running images through the network
			output = sess.run(None, {input_name: np.expand_dims(image, axis=0)})
			outputs.append(np.asarray(output)[0][0])

		# the class with the highest energy is what we choose as prediction
		predicted = np.argmax(np.array(outputs), -1)
		total += labels.size
		correct += (predicted == labels).sum().item()

	return correct / total

def main():
	parser = create_parser()
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	n_samples = args.n_samples if args.n_samples else 10000
	model_path = Path(args.model_path)
	print(model_path)

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

	onnx_accuracy = onnx_inference(str(model_path), testloader)
	print(f"ONNX Accuracy: {onnx_accuracy * 100:.2f}%")

if __name__ == "__main__":
	main()
