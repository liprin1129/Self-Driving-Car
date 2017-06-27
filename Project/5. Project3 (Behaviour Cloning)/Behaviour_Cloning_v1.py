# -*- coding: utf-8 -*-
"""
Pandas
"""

import csv
import cv2
import numpy as np
import pandas as pd

log_csv = pd.read_csv('./data/driving_log_change.csv')
'''
log_csv = pd.read_csv('./data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
#print(log_csv['center'].loc[0])
old = log_csv['center'].loc[0][:132]
print(old)
new = ""

def change_name(data):
    #print(data)
    return data.replace(old, new)
    #data['left'].replace(old, new)
    #data['right'].replace(old, new)

log_csv['center'] = log_csv['center'].apply(change_name)
log_csv['left'] = log_csv['left'].apply(change_name)
log_csv['right'] = log_csv['right'].apply(change_name)

#print(log_csv['right'].loc[1])

log_csv.to_csv('driving_log_change.csv')
'''

images = []
measurements = []

for line in log_csv.iterrows():
    #source_path = line[0]
    #filename = source_path.split('/')[-1]
    #print(line[1][1])

    current_path = './data/IMG/' + line[1][1]
    image = cv2.imread(current_path)
    images.append(image)
    
    ## problem below
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print(np.shape(X_train))
print(np.shape(y_train))
print(y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
