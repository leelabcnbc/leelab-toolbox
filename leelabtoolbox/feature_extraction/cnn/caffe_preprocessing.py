from __future__ import division, absolute_import, print_function, unicode_literals

from copy import deepcopy

import numpy as np

try:
    import caffe
except ImportError:
    caffe = None

# load the stat ImageNet image (as distributed with Caffe) for subtraction
from .generic_preprocessing import caffe_mu_bgr as imagenet_mu

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


def transform_batch(transformer, data, input_blob, unsafe=False):
    # use x.copy(), since Caffe can potentially overwrite data.
    # if you won't have channel swap set...
    return np.asarray([transformer.preprocess(input_blob, x if unsafe else x.copy()) for x in data])
