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
camera = jetson.utils.videoSource(1280, 720,"csi://0")

'''
zadaci:
	važno:
		1.Nisam testirao dio s klasifikacijom.
	
		2. crop slike daje exception 
	
	2.umijesto camera source od jetson utilsa koristiti opencv jer je sporo i radi probleme
	Slike koje sada dobivam stalno treba pretvarati u numpy array
	Ne može se baš tretirati njima kao array nego ima cijela dokumentacija za rad s njima
	
	
	3. (manje bitno)Dataset za detekciju je napravljen od 40 slika lista na mom stolu, trebam jos slika uzet(isto sa stola da bude dobar dok ga testiram(kao da je stol vinograd), a u buducnosti
	na razlicitim podrucjima pogotovo vinogradu.
	
	Kad važni zadaci budu rješeni, ovo će biti donekle funkcionalan detektor bolesti lišća, nadalje ga unaprijeđujemo.
	Uz 2. i 3. će biti bolji.
	
		
	

'''
model = tensorflow.keras.models.load_model("bolesti_vinoveloze.h5")

klasifikacija = []
spremne = []

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

    					img_cropped = jetson.utils.cudaAllocMapped(width=detection.Width, height=detection.Height, format=img.format)
                			crop_roi=(detection.Left, detection.Top, detection.Right, detection.Bottom)
                			jetson.utils.cudaCrop(img, img_cropped, crop_roi)

                			img_cropped = jetson.utils.cudaToNumpy(img_cropped)
                			img_cropped = img_cropped/255.0
                			#img_cropped = img_cropped.reshape(-1, 256, 256, 3)
                			spremne.append(img_cropped)

                			klasifikacija.pop(klasifikacija.index([detection, img]))




                		except Exception as e:
                			print(e)
               			 try:
                			spremne = np.array(spremne).reshape(-1, 256, 256, 3)
                			model.predict(spremne)

                		except:
                			print("greška prilikom klasifikacije")


    				


            


class Detekcija(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.start()

    def run(self):
        sigurne = 0
        
        while True:

            img, width, height = camera.Capture()
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
