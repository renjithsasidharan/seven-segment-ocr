import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import os.path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='test/1.png')
parser.add_argument('--model', type=str, default='meeter_rec_float16.tflite')

alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)

def prepare_input(image_path):
  input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  input_data = cv2.resize(input_data, (200, 31))
  input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 3)
  input_data = input_data.astype('float32')/255
  return input_data

def predict(image_path, model_path):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_data = prepare_input(image_path)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output = interpreter.get_tensor(output_details[0]['index'])
  return output

def main():
    args = parser.parse_args()

    if not os.path.isfile(args.image):
      print(f'{args.image} does not exist')
      sys.exit()

    result = predict(args.image, args.model)
    text = "".join(alphabet[index] for index in result[0] if index not in [blank_index, -1])
    print(f'Extracted text: {text}')

if __name__=="__main__":
    main()