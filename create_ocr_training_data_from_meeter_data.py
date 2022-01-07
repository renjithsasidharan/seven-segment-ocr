import json
from PIL import Image
from pathlib import Path
import numpy as np
import uuid

input_dir = 'data/tmp/'
output_dir = 'data/ocr_training_data/'

Path(output_dir).mkdir(parents=True, exist_ok=True) 

with open(input_dir + 'Label.txt') as file:
  labels = file.readlines()
  labels = [labels.rstrip() for labels in labels]

with open(output_dir + 'gt.txt', 'w') as f:
  for label in labels:
    file_name = label.split('\t')[0]
    print(file_name)
    gts = json.loads(label.split('\t')[1])
    img = Image.open(input_dir + Path(file_name).name)

    for gt in gts:
      text = gt['transcription']
      points = np.array(gt['points'])
      x_min = min(points[:,0])
      y_min = min(points[:,1])
      x_max = max(points[:,0])
      y_max = max(points[:,1])

      gt_file_name = str(uuid.uuid4()) + '.png'
      copped_img = img.crop((x_min, y_min, x_max, y_max))
      resized_img = copped_img.resize((200, 31), Image.LANCZOS)
      resized_img.save(output_dir + gt_file_name, 'PNG')

      f.write(gt_file_name + '\t' + text + '\n')