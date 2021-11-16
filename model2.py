# Import the required libraries
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.utils import to_categorical

import os
import cv2
import numpy as np

import random


dirtest = 'dataset/test'
dirtrain = 'dataset/train'



categories= ["Black_rot","Esca_(Black_Measles)","Healthy","Leaf_blight_(Isariopsis_Leaf_Spot)"]


training_data = []
def create_training_data():
    count = []
    for c in categories:
        path = os.path.join(dirtrain,c)
        class_num = categories.index(c)
        c = 0
        for i in os.listdir(path):
            c = c+1
            try:
                img_array = cv2.imread(os.path.join(path,i))
                training_data.append([img_array,class_num])
            except Exception as e:
                pass
        count.append(c)
    return count
count_train=create_training_data()

testing_data = []
def create_testing_data():
    count=[]
    for c in categories:
        path = os.path.join(dirtest,c)
        class_num = categories.index(c)
        c = 0
        for i in os.listdir(path):
            c = c+1
            try:
                img_array = cv2.imread(os.path.join(path,i))
                testing_data.append([img_array,class_num])
            except Exception as e:
                pass
        count.append(c)
    return count
count_test=create_testing_data()

random.shuffle(training_data)
random.shuffle(testing_data)

x_train = []
y_train = []
x_test = []
y_test = []

for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
x_train=np.array(x_train).reshape(-1,256,256,3)

for features, label in testing_data:
    x_test.append(features)
    y_test.append(label)
x_test = np.array(x_test).reshape(-1,256,256,3)

model = Sequential()
model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=(256,256,3),activation='relu'))
model.add(layers.Conv2D(32,(3,3),activation='relu'))


model.add(layers.MaxPool2D(pool_size=(8,8)))

model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(layers.Conv2D(32,(3,3),activation='relu'))

model.add(layers.MaxPool2D(pool_size=(8,8)))

model.add(Activation('relu'))

model.add(Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(4,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'rmsprop',metrics = ['accuracy'])

model.summary()

y_train_cat = to_categorical(y_train,4)
y_test_cat = to_categorical(y_test,4)

history = model.fit(x_train, y_train_cat, batch_size=32, epochs=10, verbose=1, validation_split=0.15, shuffle=True)

model.save("leaf_disease_model.h5")
new_model = tf.keras.models.load_model("leaf_disease_model.h5")

#loss and accuracy of the model
loss, acc = new_model.evaluate(x_test,y_test_cat, verbose=2)
print(acc)

