## Text recognition for seven segment display using [keras-ocr](https://github.com/faustomorales/keras-ocr)

To run prediction on an image using tensorlfow lite
```console
python predict.py --image test/1.png --model meeter_rec_float16.tflite
```

Here are some results of ocr extraction:

![](results/1.jpg) ![](results/2.jpg) ![](results/3.jpg)
![](results/4.jpg) ![](results/5.jpg) ![](results/6.jpg)

Confusion matrix

![](results/cm.jpg)