#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:27:02 2017

@author: prudhvi
"""

"""
The program trains the CNN
on the Jaffe training set.

The number of iterations can be increased, 
but as default, this runs for 5 iterations on a 
20% divided validation set. 

We trained this for around 400 iterations, 
20 epochs at a time. Each epoch takes 75 seconds

Arguments:

first : Number of epochs
"""

import numpy as np
import os
import sys
from PIL import Image
np.random.rand(2)
#from keras.layers import Conv1D
files = os.listdir('aug_data_64_by_48')

n_iter = int(sys.argv[1])
tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

def targets(filename):
    targets = []
    for f in filename:
        if tag_list[0] in f:
            targets.append(0)
        if tag_list[1] in f:
            targets.append(1)
        if tag_list[2] in f:
            targets.append(2)
        if tag_list[3] in f:
            targets.append(3)
        if tag_list[4] in f:
            targets.append(4)
        if tag_list[5] in f:
            targets.append(5)
        if tag_list[6] in f:
            targets.append(6)
    return np.array(targets)


def data(filename):
    train_images = []
    for f in filename:
        current = f
        train_images.append(np.array(Image.open('aug_data_64_by_48/'+current).getdata()))    
    return np.array(train_images)

y = targets(files)
print "Fetching Data. Please wait......"
x = data(files)
print "Fetching Complete."


x = np.reshape(x, (3408, 48, 48, 1))
#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =124)
from keras.utils import np_utils
y = np_utils.to_categorical(y)
#y_test = np_utils.to_categorical(y_test)
#x_train = np.reshape(x_train, (2556, 48, 48, 1))
#x_test = np.reshape(x_test, (852, 48, 48, 1))
#print x.shape
#print y.shape
x_train = x[0:2556, :, :, :]
x_test = x[2556:, :, :, :]
y_train = y[0:2556, :]
y_test = y[2556:, :]

#print "\nConfusion Matrix\n"
#print confusion_matrix(y_test, predictions)
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()    
model.add(Conv2D(10, 5, 5, activation = 'relu', input_shape = x.shape[1:]))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, 5, 5, activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

#model.add(Conv2D(10, 3, 3, activation = 'relu'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))


#ada = optimizers.adam(lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay= 0)
#ada = optimizers.adam(lr = 0.005)
model.compile(optimizer= 'adam' , loss = 'categorical_crossentropy',
              metrics= ['accuracy'])

model.fit(x_train, y_train, batch_size= 100, nb_epoch= n_iter, validation_split=0.2)
model.save("CNN_Jaffe_model_" + str(n_iter) + "_epoch.h5")
