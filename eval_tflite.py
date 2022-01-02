import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import time

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir', type=str, default='data/ocr_training_data')


alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)

def run_tflite_model(image_path, interpreter):
    input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_data = cv2.resize(input_data, (200, 31))
    input_data = input_data[np.newaxis]
    input_data = np.expand_dims(input_data, 3)
    input_data = input_data.astype('float32')/255
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def evaluate(quantization, data_dir):
  gt_file = f'{data_dir}/gt.txt'
  model_path = f'meeter_rec_{quantization}.tflite'

  with open(gt_file) as file:
      lines = file.read().splitlines()
      train_labels = [(data_dir + '/' + line.split('\t')[0], line.split('\t')[1]) for line in lines]

  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  results = []
  with tqdm(total=len(train_labels)) as pbar:
    for image_path, actual in train_labels:
      tflite_output = run_tflite_model(image_path, interpreter)
      prediction = "".join(alphabet[index] for index in tflite_output[0] if index not in [blank_index, -1])
      results.append(prediction)
      pbar.update()

  return train_labels, results


def main():
    args = parser.parse_args()
    metrics = dict()
    for quantization in ['float16']:
      print(f'Evaluating {quantization} model')

      t0 = time.time()
      train_labels, results = evaluate(quantization, args.eval_dir)
      t1 = time.time()

      results_boolean =[actual == result for (_, actual), result in zip(train_labels, results)]
      accuracy = round(100*sum(results_boolean)/len(results_boolean), 2)
      avg_inference_time = round((t1-t0)/len(results_boolean), 2)
      metrics[quantization] =  {
          'accuracy': accuracy,
          'avg_inference_time': avg_inference_time
      }

    for k, v in metrics.items():
      print(f'{k}')
      print(f'    Accuracy: {v["accuracy"]}')
      print(f'    Average inference time: {v["avg_inference_time"]} sec')

if __name__=="__main__":
    main()