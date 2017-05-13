#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:07:48 2017

@author: prudhvi
"""

"""

Takes a dataset of 64 x 64 sized images
and returns 48 x 48 patches of each image

Arguments:

First: train_files.txt (or) test_files.txt

Be sure to create a new directory 
of the name given below
before executing this program
"""


import numpy as np
from PIL import Image
import os
from scipy.misc import imsave
import sys

files = sys.argv[1]
filenames = open(files, 'r').read().splitlines()

for i in range(0, len(filenames)):
    print i
    current = filenames[i]  
    if current != '.DS_Store':
        image = np.array((Image.open('resized_JAFFE_data_64_by_64/' + current)).getdata())
        image = np.reshape(image, (64, 64))
        for i in range(0, 16):
            if files == 'train_files.txt':
                imsave('aug_data_64_by_48/' + str(i) + '_' + current, image[i:i+48, i:i+48])  
            	#imsave('aug_train/' + str(i) + '_' + current, image[i:i+48, i:i+48])  

            else:
                #imsave('aug_test/' + str(i) + '_' + current, image[i:i+48, i:i+48])

	        imsave('aug_test_data_64_by_48/' + str(i) + '_' + current, image[i:i+48, i:i+48])     

