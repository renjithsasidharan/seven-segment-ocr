## Text recognition for seven segment display using [keras-ocr](https://github.com/faustomorales/keras-ocr)

#### Requirements
```
tensorflow==2.7.0
imgaug
tqdm
opencv-python
matplotlib
sklearn
```

#### Training
Training data is present in `data/ocr_training_data`
```console
data/ocr_training_data
├── 00498afd-2925-45ef-bfb3-8c55204ace42.png
├── 011fd0b2-837c-424c-a71b-a2a92305a532.png
├── 04976571-ac6b-425f-9474-3b95a4fb9613.png
├── 068e4ff1-acf0-43e7-b052-7a59a9c3f87b.png
```
Ground truth file is `data/ocr_training_data/gt.txt`

Train the model using jupyter notebook `keras_ocr_7_seg.ipynb`

#### Evaluation
To run prediction on an image using tensorlfow lite
```console
python predict.py --image test/1.png --model meeter_rec_float16.tflite
```

To run predictions on all images in directory
```console
python eval_tflite.py --eval_dir /data/ocr_training_data
```

Here are some results of ocr extraction:

![](results/1.jpg) ![](results/2.jpg) ![](results/3.jpg)
![](results/4.jpg) ![](results/5.jpg) ![](results/6.jpg)

Confusion matrix

![](results/cm.jpg)