import numpy as np
import pickle
import os
from PIL import Image
from scipy.spatial.distance import cosine, euclidean, correlation
import operator
import sys

"""

The program is used for testing the 
shallow autoencoder network.

This takes three arguments.

first: dataset (jaffe for JAFFE test set, and lfw for LFW data set)
second: Number of Hidden Layers (300/500)
third: Number of iterations for each example (100 was giving the best results)

Once it takes all the images,
it converts them into 300/500 dimensional 
space and uses cosine distance as a metric 
to find the distance between two emotions.

"""
np.random.rand(30)
c = sys.argv[1]
hu = sys.argv[2]
if c == 'lfw':
    test = os.listdir('lfw2/')
elif c == 'jaffe':
    test = open('test_files.txt', 'r').read().split()

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras import backend as K
#data = np.array(list(Image.open('resized_JAFFE_data_64_by_64/' +test[0]).getdata()))
train_dict = pickle.load(open('one_layer_ae_' + str(hu) + '_hn_rep/trained_' +str(hu)+ '_representations.txt', 'rb'))

tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']
targets_list = []
for t in test:
    for tag in tag_list:
	if tag in t:
	    targets_list.append(tag)


#print len(targets_list)
#print targets_list
#print data.shape
#print test[0]
#print e

hidden_units = int(hu)
n_iter = int(sys.argv[3])
#data = np.reshape(data, (1, 4096))
np.random.rand(30)
inputs = Input(shape = (4096, ))
encoder = Dense(hidden_units)(inputs)
decoder = Dense(4096)(encoder)
model = Model(input = inputs, output = decoder)

model.compile(optimizer = 'adam', loss = 'mse')

a = []
b = []
for t in range(len(test)):
    print t
    if c == 'lfw':
        im = Image.open('lfw2/' + test[t])
        im = im.convert('L')
        data=  im.resize((64, 64))
        data = np.array(data.getdata())
    elif c == 'jaffe':
        data = np.array(list(Image.open('resized_JAFFE_data_64_by_64/' +test[t]).getdata()))
    data = np.reshape(data, (1, 4096))
    model.fit(data, data, nb_epoch = n_iter)

#from keras import backend as K
    encoder_func = K.function([model.layers[0].input], [model.layers[1].output])
    encoder_output = np.array(encoder_func([data]))

    hidden_rep = encoder_output[0, 0, :]

#print hidden_rep

#train_dict = pickle.load(open('one_layer_ae_300_hn_rep/trained_300_representations.txt', 'rb'))

    new_dict = {}

    for p in train_dict:
    	new_dict[p] =  cosine(hidden_rep,train_dict[p])

    order  = sorted(new_dict.items(), key=operator.itemgetter(1))

    a.append((order[0][0], order[1][0], order[2][0]))

#print a
#print targets_list

a = np.array(a)
targets_list = np.array(targets_list)

count = 0
for t in range(len(targets_list)):
    if targets_list[t] in a[t]:
	count +=1

print "Number of Correct Predictions (Top 2): " + str(count)
print "Total Number of Images: " +str(len(targets_list))

print "Accuracy : " + str((1.0 * count)/len(targets_list))
#print np.sum(a == targets_list)
#print len(targets_list)
