'''
References:

1)https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
'''

"""

The program is for the shallow 
autoencoder network.

The strucutre of this autoencoder is 
4096 - 300/500 - 4096. 

Arguments:

First: Emotion - [AN, SA, SU, NE, FE, DI, NE]
Second: Number of Iterations
Third: Number of Hidden Units - 300 (or) 500

For each given emotion, this program creates a condensed
representation that is said to capture some unique
information about that particular emotion.

The loads the train_files.txt file, which consists of 
the filenames of the training data and then collects each 
image from the directory.

We used pickle to save all these hidden representations.

MAKE SURE TO RUN THE AUTOENCODER APPENDER WITH CORRECT
ARGUMENTS BEFORE TESTING
"""
import pickle
import sys
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
import numpy as np
from scipy.misc import imread, imsave
from PIL import Image
import os
from sklearn.model_selection import train_test_split
np.random.seed(30)

new_files = []
#files = os.listdir('resized_JAFFE_data_64_by_64/')
files = open('train_files.txt', 'r').read().split()

#print f
#print e
emotion = str(sys.argv[1])
n_iter = int(sys.argv[2])
hidden_units = int(sys.argv[3])
print n_iter
print emotion
#print e
for f in files:
    if emotion in f:
    	new_files.append(f)

#print new_files
print len(new_files)
print np.array(new_files).shape
#print e
#x, y = train_test_split(files, test_size = 0.25, random_state = 12)
#print len(x)
#print len(y)
#print y
#print e
data = []

#print new_files
#print e
for f in new_files:
    print f
    im = np.array(list(Image.open('resized_JAFFE_data_64_by_64/' +f).getdata()))#imread("/resized_JAFFE_data_64_by_64/TM.NE1.177.tiff")
    #data = np.concatentate((data, im))
    data.append(im)
#img2 = np.reshape(np.array(im), (1, 4096))
#data = open('test_data.txt', 'r').read().split()
#data = np.array(data).astype(np.int)
#data = np.reshape(data, (1, 4096))

print np.array(data).shape
#print img2.shape
#print data.shape
#print np.concatenate((data, img2)).shape
#print data.shape
#print e
#data = np.concatenate((data, img2))

data = np.array(data)
inputs = Input(shape = (4096, ))
encoder = Dense(hidden_units)(inputs)
decoder = Dense(4096)(encoder)
model = Model(input = inputs, output = decoder)

model.compile(optimizer = 'adam', loss = 'mse')

model.fit(data, data, nb_epoch = n_iter)

from keras import backend as K
encoder_func = K.function([model.layers[0].input], [model.layers[1].output])
encoder_output = np.array(encoder_func([data]))

hidden_rep = encoder_output[0, 0, :]

reps = open('one_layer_ae_'+str(hidden_units)+'_hn_rep/' + emotion + '_' + str(hidden_units) + '_rep.txt', 'wb')
hidden_rep = np.array(hidden_rep)

pickle.dump(hidden_rep, reps)
reps.close()

weights = np.array(((model.layers[1]).get_weights())[0])

new_file = open('weights.txt', 'wb')
pickle.dump(weights, new_file)
new_file.close()

#print data
val = model.predict(data)
#print emotion
#print np.max(hidden_rep), np.min(hidden_rep)
#print hidden_rep.shape
#print val.shape
#print data.shape
#print e
#imsave('original.jpg', np.reshape(data[0, :], (64, 64)))
#imsave('predicted.jpg', np.reshape(val[0, :], (64, 64)))
#print np.array(model.predict(data)).shape



#new = open('AN_10_rep.txt', 'rb')
#print pickle.load(new)
'''
model = Sequential()
model.add(Dense(30, input_dim = 4096, init = 'uniform'))
model.add(Activation('linear'))
model.add(Dense(4096, init = data[0]))
model.add(Activation('linear'))
'''
