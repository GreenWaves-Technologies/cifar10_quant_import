from tqdm import tqdm
import numpy as np
import pathlib
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import argparse
import argcomplete

def create_parser():
	# create the top-level parser
	parser = argparse.ArgumentParser(prog='test_gapflow')

	parser.add_argument('--model_path', type=str, required=True,
						help='path to model to test')
	parser.add_argument('--n_samples', type=int, default=10000,
						help='number of samples to test on, default: all')
	return parser

# Helper function to run inference on a TFLite model
def test_tflite_model(tflite_file, test_images, test_labels):
	# Initialize the interpreter
	interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]

	predictions = np.zeros((len(test_images),), dtype=int)
	for i, (test_image, test_label) in enumerate(tqdm(zip(test_images, test_labels), total=len(test_labels))):
		# Check if the input type is quantized, then rescale input data to uint8
		if input_details['dtype'] == np.uint8:
			input_scale, input_zero_point = input_details["quantization"]
			test_image = test_image / input_scale + input_zero_point

		test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
		interpreter.set_tensor(input_details["index"], test_image)
		interpreter.invoke()
		output = interpreter.get_tensor(output_details["index"])[0]

		predictions[i] = output.argmax()

	test_labels_not_one_hot = np.argmax(test_labels, 1)
	accuracy = (np.sum(test_labels_not_one_hot == predictions) * 100) / len(test_images)
	return accuracy


def main():
	parser = create_parser()
	argcomplete.autocomplete(parser)
	args = parser.parse_args()

	n_samples = args.n_samples if args.n_samples else 10000
	model_path = Path(args.model_path)
	print(model_path)

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

	tflite_model_file = pathlib.Path(model_path)
	fp32_accuracy = test_tflite_model(tflite_model_file, test_images, test_labels)
	print(f"Float model accuracy: {fp32_accuracy}%")

	tflite_quant_model_file = pathlib.Path("model/cifar10_model_uint8.tflite")
	quant_accuracy = test_tflite_model(tflite_quant_model_file, test_images[:n_samples], test_labels[:n_samples])
	print(f"Quantized model accuracy: {quant_accuracy}%")


if __name__ == "__main__":
	main()
