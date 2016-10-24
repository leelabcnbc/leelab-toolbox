from __future__ import division, absolute_import, print_function, unicode_literals

import os.path
from copy import deepcopy

import numpy as np

try:
    import caffe
except ImportError:
    pass

from . import dir_dict

# load the mean ImageNet image (as distributed with Caffe) for subtraction
try:
    imagenet_original = np.load(os.path.join(dir_dict['caffe_root'], 'imagenet', 'ilsvrc_2012_mean.npy'))
    imagenet_mu = imagenet_original.mean(1).mean(1)  # this one is BGR
except OSError:
    imagenet_original = None
    imagenet_mu = np.zeros(3, dtype=np.float64)  # match data type of one in caffe.
imagenet_mu_rgb = imagenet_mu[::-1]


def create_transformer(input_blob, input_blob_shape, scale=255, mu=None):
    if mu is None:
        mu = imagenet_mu  # default mean for most of Caffe models.
    # get transformer
    # use deep copy to avoid tricky bugs for reference.
    transformer = caffe.io.Transformer({input_blob: deepcopy(input_blob_shape)})
    transformer.set_transpose(input_blob, (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean(input_blob, deepcopy(mu))  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale(input_blob, scale)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap(input_blob, (2, 1, 0))  # swap channels from RGB to BGR
    return transformer


def transform_batch(transformer, data, input_blob):
    # use x.copy(), since Caffe can potentially overwrite data.
    # if you won't have channel swap set...
    return np.asarray([transformer.preprocess(input_blob, x.copy()) for x in data])
