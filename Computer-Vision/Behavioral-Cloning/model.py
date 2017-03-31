#!/usr/bin/env python3

import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Cropping2D
import keras.backend.tensorflow_backend as K

samples = []
for batch in os.listdir('./data/'):
    with open('./data/%s/driving_log.csv' %batch) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
print("Number of samples = %s" %len(samples))

X_train = np.zeros((32530 * 2, 160, 320, 3))
y_train = np.zeros((32530 * 2))

images = []
measurements = []
for batch in os.listdir('./data/'):
    with open('./data/%s/driving_log.csv' %batch) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            correction = 0.18
            
            steering_center = float(line[3])
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            center_file = line[0].split('/')[-1]
            current_path_center = './data/%s/IMG/' %batch + center_file
            center_image = cv2.imread(current_path_center) 
            images.append(center_image)
            measurements.append(steering_center)
            images.append(cv2.flip(center_image, 1))
            measurements.append(steering_center * -1.0)
            
            left_file = line[1].split('/')[-1]
            current_path_left = './data/%s/IMG/' %batch + left_file
            left_image = cv2.imread(current_path_left)
            images.append(left_image)
            measurements.append(steering_left)
            images.append(cv2.flip(left_image, 1))
            measurements.append(steering_left * -1.0)
            
            right_file = line[2].split('/')[-1]
            current_path_right = './data/%s/IMG/' %batch + right_file
            right_image = cv2.imread(current_path_right)
            images.append(right_image)
            measurements.append(steering_right)
            images.append(cv2.flip(right_image, 1))
            measurements.append(steering_right * -1.0)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Convolution2D(24, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(48, 3, 3, border_mode='same', subsample=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 2, 2, border_mode='same', activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.20, shuffle=True, nb_epoch=5)

model.save('model.h5')
