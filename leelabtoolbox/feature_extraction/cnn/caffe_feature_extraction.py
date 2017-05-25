from __future__ import division, absolute_import, print_function, unicode_literals
import numpy as np
from collections import defaultdict
import time


def reshape_blobs(net, input_blobs, batch_size):
    # reshape the net for input blobs.
    for in_blob in input_blobs:
        shape_old = net.blobs[in_blob].data.shape
        assert len(shape_old) == 4
        if shape_old[0] != batch_size:
            print('do reshape!')
            net.blobs[in_blob].reshape(batch_size, *shape_old[1:])


def _extract_features_input_check(net, data_this_caffe, input_blobs,
                                  blobs_to_extract):
    assert isinstance(data_this_caffe, list), 'you must pass a list of inputs, even for single input'
    if input_blobs is None:
        input_blobs = [net.inputs[0]]  # only take one, as in caffe's classifier.
    if blobs_to_extract is None:
        blobs_to_extract = set(net.blobs.keys()) - set(net.inputs)
    # print('blobs to extract: {}'.format(blobs_to_extract))

    # check each data has same size
    num_image = len(data_this_caffe[0])
    for data_this_caffe_this in data_this_caffe:
        assert len(data_this_caffe_this) == num_image

    return input_blobs, blobs_to_extract, num_image


def _extract_features_one_loop(feature_dict,
                               net, data_this_caffe, input_blobs, blobs_to_extract, batch_size,
                               num_image, slice_dict, startidx, unsafe=False, raw_dim=False, last_layer=None):
    slice_this_time = slice(startidx, min(num_image, startidx + batch_size))
    slice_out_this_time = slice(0, slice_this_time.stop - slice_this_time.start)

    # set data.
    for idx, in_blob in enumerate(input_blobs):
        net.blobs[in_blob].data[slice_out_this_time] = data_this_caffe[idx][slice_this_time]

    # then forward.
    net.forward(end=last_layer)

    for blob in blobs_to_extract:
        slice_r, slice_c = slice_dict[blob]
        blob_raw = net.blobs[blob].data[slice_out_this_time]
        if not raw_dim:
            if blob_raw.ndim == 2:  # this can be the case for full connection layers.
                blob_raw = blob_raw[:, :, np.newaxis, np.newaxis]
            assert blob_raw.ndim == 4
            # this copy is important... otherwise there can be issues.
            # slice only works with not raw dim.
        if blob_raw.ndim == 4:
            data_this_to_use = blob_raw[:, :, slice_r, slice_c]
        else:
            data_this_to_use = blob_raw

        if not unsafe:
            data_this_to_use = data_this_to_use.copy()
        feature_dict[blob].append(data_this_to_use)


def extract_features(net, data_this_caffe, input_blobs=None,
                     blobs_to_extract=None, batch_size=50, slice_dict=None, unsafe=False,
                     raw_dim=False, last_layer=None):
    if slice_dict is None:
        slice_dict = defaultdict(lambda: (slice(None, None), slice(None, None)))

    input_blobs, blobs_to_extract, num_image = _extract_features_input_check(net, data_this_caffe,
                                                                             input_blobs, blobs_to_extract)
    reshape_blobs(net, input_blobs, batch_size)

    feature_dict = defaultdict(list)
    if unsafe:
        assert batch_size >= num_image
    # then do the actual computation
    # start_t_f = time.time()
    for startidx in range(0, num_image, batch_size):
        _extract_features_one_loop(feature_dict,
                                   net, data_this_caffe, input_blobs, blobs_to_extract, batch_size,
                                   num_image, slice_dict, startidx, unsafe=unsafe, raw_dim=raw_dim,
                                   last_layer=last_layer)
    # end_t_f = time.time()
    # print('actual extraction time {}'.format(end_t_f-start_t_f))
    # start_t_f1 = time.time()
    for blob_out in feature_dict:
        if not unsafe:
            feature_dict[blob_out] = np.concatenate(feature_dict[blob_out], axis=0)
        else:
            feature_dict[blob_out] = feature_dict[blob_out][0]
    # end_t_f1 = time.time()
    # print('post processing time {}'.format(end_t_f1 - start_t_f1))

    return feature_dict
