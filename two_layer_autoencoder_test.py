import numpy as np
import pickle
import os
from PIL import Image
from scipy.spatial.distance import cosine, euclidean, correlation
import operator
import sys
"""

The program is used for testing the 
two layer dense autoencoder network.

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

#np.random.rand(30)
#test = os.listdir('lfw2/')
#test = open('test_files.txt', 'r').read().split()

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras import backend as K
#data = np.array(list(Image.open('resized_JAFFE_data_64_by_64/' +test[0]).getdata()))
train_dict = pickle.load(open('two_layer_ae_' +str(hu) + '_hn_rep/trained_'+str(hu) +'_representations.txt', 'rb'))

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
'''
n_iter = 40
#data = np.reshape(data, (1, 4096))
np.random.rand(30)
inputs = Input(shape = (4096, ))
encoder = Dense(hidden_units)(inputs)
decoder = Dense(4096)(encoder)
'''
h2 = 2800
#hidden_units = 300
n_iter = int(sys.argv[3])
#data = np.array(data)
inputs = Input(shape = (4096, ))
#encoder = Dense(hidden_units)(inputs)
encoder = Dense(h2)(inputs)
encoder_2 = Dense(hidden_units)(encoder)
decoder_2 = Dense(h2)(encoder_2)
decoder = Dense(4096)(decoder_2)
#model = Model(input = inputs, output = decoder)


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
    encoder_func = K.function([model.layers[1].input], [model.layers[2].output])
    encoder_output = np.array(encoder_func([data]))

    hidden_rep = encoder_output[0, 0, :]

#print hidden_rep

#train_dict = pickle.load(open('one_layer_ae_300_hn_rep/trained_300_representations.txt', 'rb'))

    new_dict = {}

    for p in train_dict:
    	new_dict[p] =  cosine(hidden_rep,train_dict[p])

    order  = sorted(new_dict.items(), key=operator.itemgetter(1), reverse = True)
    order_2 = sorted(new_dict.items(), key=operator.itemgetter(1))
    #a.append((order_2[0][0], order_2[1][0]))#, order[2][0]))
    b.append((order_2[0][0], order_2[1][0], order_2[2][0]))

#print a
#print b
#print targets_list

#a = np.array(a)
b = np.array(b)
targets_list = np.array(targets_list)

#count_a = 0
count_b = 0
for t in range(len(targets_list)):
    #if targets_list[t] in a[t]:
    #	count_a +=1
    if targets_list[t] in b[t]:
	count_b +=1

#print count_a

print "Number of Correct Predictions (Top 2): " + str(count_b)
print "Total Number of Predictions: " +str(len(targets_list))

print "Accuracy : " + str((1.0 * count_b)/len(targets_list))
#print np.sum(a == targets_list)
#print len(targets_list)
