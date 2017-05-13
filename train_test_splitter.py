import os
from sklearn.model_selection import train_test_split
import numpy as np

"""

Splits files for autoencoder modules

"""
files = os.listdir('resized_JAFFE_data_64_by_64/')

train_data, test_data = train_test_split(files, test_size  = 0.25, \
			random_state = 123)

def counter(tag, files):
    count = 0
    for f in files:
	if tag in f:
	    count +=1
    return count

tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

for t in tag_list:
    print t, counter(t, train_data), counter(t, test_data)

train_file = open('train_files.txt', 'w')
for t in train_data:
    train_file.write(t + '\n')
train_file.close()

test_file = open('test_files.txt', 'w')
for t in test_data:
    test_file.write(t + '\n')
test_file.close()
