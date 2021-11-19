import jetson.inference
import jetson.utils

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from threading import Thread

import tensorflow as tf
from tensorflow import keras


path = '/home/robotika/jetson-inference/python/training/detection/ssd/models/test3'


net = jetson.inference.detectNet(argv=['--model={}'.format(path + "/ssd-mobilenet.onnx"), '--labels={}'.format(path + "/labels.txt"), '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'], threshold=0.5)
camera = jetson.utils.videoSource("csi://0")


model = tf.keras.models.load_model("bolesti.h5")

klasifikacija = []
lst = []
gotove =  []


class Procesiranje(Thread):
    def __init__(self):
	
        Thread.__init__(self)
        self.daemon = True
        self.start()

    def run(self):
    	
        while True:
            if(len(klasifikacija)):

                for detection, img in klasifikacija:
                    try:
                        img_arry = jetson.utils.cudaToNumpy(img)
                        img_arry = cv2.resize(img_arry, (256, 256))
                        klasifikacija.pop(klasifikacija.index([detection, img]))
                        lst.append(img_arry)
			
                    except Exception as e:
                        print(e)

                try:
                    ary = numpy.array(lst).reshape(-1, 256, 256, 3)
                    prediction = model.predict(ary)

                    for i in range(len(prediction)):
                        b = np.argmax(prediction[i])
                        cv2.imshow(str(b), ary[i])
                        cv2.waitKey(0)
                        gotove.append([b, ary[i]])
                        
		    lst.clear()

                except Exception as e:
                    print(e)

class Detekcija(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.start()

    def run(self):
        sigurne = 0
        
        while True:

            img = camera.Capture()
            detections = net.Detect(img, overlay='none')
    
            for detection in detections:
                if detection.Confidence>0.8:
                    sigurne+=1
                    klasifikacija.append([detection, img])
    
            print("Detekcije visoke vjerovatnosti: ", sigurne, "od ukupnih ", len(detections))
            sigurne= 0
            time.sleep(1)
    

Procesiranje()
Detekcija()

while True:
    pass
