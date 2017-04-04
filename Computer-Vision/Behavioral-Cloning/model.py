#!/usr/bin/env python3

import os
import csv

import cv2
import numpy as np
import sklearn

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Cropping2D, AveragePooling2D
import keras.backend.tensorflow_backend as K


def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.18
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # For each sample read the center, left, and right image data
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Reading center image
                batch = batch_sample[-1]
                name = './data/%s/IMG/' %batch + batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # Flip center image
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)
                
                # Reading left image
                name = './data/%s/IMG/' %batch + batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)
                
                # Flip left image
                images.append(cv2.flip(left_image, 1))
                angles.append(left_angle * -1.0)
                
                # Reading right image
                name = './data/%s/IMG/' %batch + batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)
                
                # Flip right image
                images.append(cv2.flip(right_image, 1))
                angles.append(right_angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def main():
    # Data has been stored in mulitple directories under 'data' directory
    # So loop through all the sub-directories of the 'data' directory
    # and read every sample fromo each driving_log.csv file
    samples, batch = [], []
    for batch in os.listdir('./data/'):
        with open('./data/%s/driving_log.csv' %batch) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                line.append(batch)
                samples.append(line)
    
    # Split data into training set = 70% and validation set = 30%
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.3)
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    
    # Below is the model architecture
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
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    
    # Train the model
    model.fit_generator(train_generator,
                        samples_per_epoch= len(train_samples) * 6,
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples) * 6,
                        nb_epoch=9)
    
    model.save('model.h5')
    
    pass

if __name__ == "__main__":
    main()
