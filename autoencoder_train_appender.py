import numpy as np
import pickle
import os
import sys

"""

This program appends all the representations of
emotions into one file

Arguments:

first: Number of layers (one/two)
second: Number of hidden units (300/500)

PLEASE MAKE SURE TO RUN THIS BEFORE RUNNING 
THE AUTOENCODER TEST FILES
"""

layers = sys.argv[1]
hu = sys.argv[2]
#data = os.listdir('two_layer_ae_500_hn_rep/')
data = os.listdir(str(layers) + '_layer_ae_' +str(hu) +'_hn_rep/')

new_dict = {}

emotions = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

for e in emotions:
    for d in data:
	if e in d:
	   f = open(layers+'_layer_ae_'+ hu + '_hn_rep/' +d, 'rb')
	   new_dict[e] = pickle.load(f)

dict_write = open(layers + '_layer_ae_' + hu + '_hn_rep/trained_' +hu + '_representations.txt', 'wb')
pickle.dump(new_dict, dict_write)
dict_write.close()


