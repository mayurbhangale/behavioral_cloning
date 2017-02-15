import csv
import cv2

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
			source_path = line[0]
			tokens = source_path.split('/')
			filename = tokens[-1]
			local_path = "../data/IMG/" + filename
			image = cv2.imread(local_path)
			images.append(image)
	correction = 0.3
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+correction)
	measurements.append(measurement-correction)

print(len(images))
print(len(measurements))
# print(local_path)
# exit()

# augmented_images = []
# augmented_measurement = []
# for image,measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurement.append(measurement)
# 	flipped_image = cv2.flip(image, 1)
# 	flipped_measurement = float(measurement) * -1.0
# 	augmented_images.append(flipped_image)
# 	augmented_measurement.append(flipped_measurement)

import numpy as np

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) #Normalizing
model.add(Cropping2D(cropping=((70,25),(0,0)))) #Cropping upper part 70px, bottom 25px
model.add(Convolution2D(6,5,5,activation='relu')) #LeNet
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle = True, nb_epoch=3)

model.save('model.h5')
