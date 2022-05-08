#!/usr/bin/env python

import json
import csv
from pathlib import Path
from PIL import Image
import numpy as np


input_dir = 'data/annotations/'
output_dir = 'data/ocr_training_data/'


with open(output_dir + '/gt.txt', mode='r') as gt_file:
    reader = csv.reader(gt_file, delimiter='\t')
    ground_truths = {rows[0]:rows[1] for rows in reader}

with open(input_dir + 'Label.txt') as file:
  labels = file.readlines()
  labels = [labels.rstrip() for labels in labels]

for label in labels:
    file_name = label.split('\t')[0]
    print(file_name)
    annotations = json.loads(label.split('\t')[1])
    if len(annotations) > 0:
      annotation = annotations[0]
      ground_truths[Path(file_name).name] = annotation['transcription']

      img = Image.open(input_dir + Path(file_name).name)
      points = np.array(annotation['points'])
      x_min = min(points[:,0])
      y_min = min(points[:,1])
      x_max = max(points[:,0])
      y_max = max(points[:,1])

      copped_img = img.crop((x_min, y_min, x_max, y_max))
      resized_img = copped_img.resize((200, 31), Image.LANCZOS)
      resized_img.save(output_dir + Path(file_name).name, 'PNG')


with open(output_dir + 'gt.txt', 'w') as gt_file:  
    writer = csv.writer(gt_file, delimiter='\t')
    for key, value in ground_truths.items():
       writer.writerow([key, value])

