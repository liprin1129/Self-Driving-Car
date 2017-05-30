# -*- coding: utf-8 -*-
"""
Pandas
"""

import csv
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

print(np.shape(lines))
for idx, line in tqdm(enumerate(lines)):
    if idx > 0:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')
            #print(filename[-1])
            current_path = 'data/IMG/' + filename[-1]
            #print(current_path)

            image = cv2.imread(current_path)
            images.append(image)
            #print(np.shape(image))
            #print(line[3])
        correction = 0.2
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(measurement+correction)
        measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurment * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
'''
import os
import pickle

def saveToPickle(data, file_name=None, folder_path=None):
    if not os.path.isdir(folder_path):
        #print("Create \"preprocessed-data\" folder")
        os.mkdir(folder_path)
    else:
        print("folder already exist")

    file_name = folder_path + file_name
    if not os.path.exists(file_name):
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                print("Create", file_name)
        except Exception as e:
            print('Error: unable to save data to', file_name, 'because', e)
            
def loadPickle(file_name=None, folder_path=None):
    file = folder_path + file_name
    #print('Load')
    if os.path.exists(file):
        try:
            with open(file, 'rb') as f:
                return pickle.load(f)
                print("Open", file)
        except Exception as e:
            print('Error: unable to open data to', file, 'because', e)

saveToPickle(X_train, 'X_train.p', './data/')
saveToPickle(y_train, 'y_train', './data/')
'''
print(np.shape(X_train))
print(np.shape(y_train))
print(y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x /255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 6, 5, subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=2)

model.save('model.h5')
