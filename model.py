import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random


lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

#split the data into train and validation data set
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)

def process_image(path):
	'''
	process the image with the path
	'''
	filename = path.split('\\')[-1]#windows "\\" linux "/"
	current_path = './data/IMG/'+ filename
	image = cv2.imread(current_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#change color space from BGR to RGB
	return image

def generator(samples, batch_size =32):
	'''
	generator function include data augmented with multiple cameras, 
	flip images
	'''
	num_samples = len(samples)
	while 1:
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				img_center = process_image(batch_sample[0])#multiple cameras center
				img_left = process_image(batch_sample[1])#left cameras
				img_right = process_image(batch_sample[2])#right cameras

				steering_center = float(batch_sample[3])
				correction = 0.2
				steering_left = steering_center + correction#multiple cameras data modify
				steering_right = steering_center - correction

				images.extend([img_center,img_left,img_right])
				angles.extend([steering_center,steering_left,steering_right])

				images.extend([cv2.flip(img_center,1),cv2.flip(img_left,1),cv2.flip(img_right,1)])#augmented_images
				angles.extend([steering_center*(-1.0),steering_left*(-1.0),steering_right*(-1.0)])#augmented_measurements

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size =32)
validation_generator = generator(validation_samples, batch_size =32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from keras import backend as K
#bulid model wit keras
model = Sequential()
model.add(Lambda(lambda x:x /127.5-1.0,input_shape = (160,320,3)))#normalize use lambda
model.add(Cropping2D(cropping=((70,25),(0,0))))#Cropping the picture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))#Conv1 layer
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))#Conv2 layer
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))#Conv3 layer
model.add(Convolution2D(64,3,3,activation="relu"))#Conv4
model.add(Convolution2D(64,3,3,activation="relu"))#Conv5
model.add(Flatten())#Flatten layer
model.add(Dense(100))#Full connect layer
model.add(Dense(50))#Full connect layer
model.add(Dense(10))#Full connect layer
model.add(Dense(1))#Full connect layer


model.compile(loss='mse', optimizer='adam')

#Data has been augmented,so the samples should be 6 times than previous
history_object = model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples),validation_data=validation_generator, nb_val_samples=6*len(validation_samples),nb_epoch=4)

print(history_object.history.keys())

plt.switch_backend('agg')#use backed 'agg' in remote linux service
plt.plot(history_object.history['loss'])#loss data of train
plt.plot(history_object.history['val_loss'])#loss data of validation
plt.title('model mean aquared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'], loc ='upper right')
plt.savefig("history.jpg")#save figure in linux service

model.save('model.h5')#save model
K.clear_session()
exit()
