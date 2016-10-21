from __future__ import division, absolute_import, print_function

from collections import defaultdict
import os.path
from copy import deepcopy

import numpy as np

try:
    import caffe
except ImportError:
    pass

from . import dir_dict


def get_slice_dict(slice_dict, blobs_to_extract):
    # then, compute the slice dict
    if slice_dict is None:
        slice_dict = defaultdict(lambda: ((None, None), (None, None)))

    slice_dict_real = dict()

    for blob_name_to_read in blobs_to_extract:
        slice_exp_1, slice_exp_2 = slice_dict[blob_name_to_read]
        slice_r = slice(slice_exp_1[0], slice_exp_1[1])
        slice_c = slice(slice_exp_2[0], slice_exp_2[1])
        slice_dict_real[blob_name_to_read] = slice_r, slice_c
    return slice_dict_real


# load the mean ImageNet image (as distributed with Caffe) for subtraction
imagenet_mu = np.load(os.path.join(dir_dict['caffe_repo_root'],
                                   'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1)


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
    return np.asarray([transformer.preprocess(input_blob, x) for x in data])
