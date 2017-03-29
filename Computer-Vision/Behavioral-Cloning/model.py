#!/usr/bin/env python3

import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


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
