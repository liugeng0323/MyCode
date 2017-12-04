# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:14:16 2017

@author: leo
"""

import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
caffe_root = ''

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
import sys
#caffe_root = 'C:/Users/leo/pycaffe/'     # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root )

import caffe
import os
if os.path.isfile('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print ('CaffeNet found.')
else:
    print ('Downloading pre-trained CaffeNet model...')
caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
mu = np.load( 'models/bvlc_reference_caffenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print( 'mean-subtracted values:', zip('BGR', mu))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
image = caffe.io.load_image('models/bvlc_reference_caffenet/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print ('predicted class is:', output_prob.argmax())
labels_file = caffe_root + 'models/bvlc_reference_caffenet/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')

print ('output label:', labels[output_prob.argmax()])
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print ('probabilities and labels:')
print(list(zip(output_prob[top_inds], labels[top_inds])))