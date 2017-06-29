from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict


def _create_blob_info_dict(info_dict_raw):
    info_dict = OrderedDict()
    for layer, stride, kernelsize, pad in info_dict_raw:
        info_dict[layer] = {
            'stride': stride,
            'kernelsize': kernelsize,
            'pad': pad
        }
    return info_dict


blob_info = OrderedDict()
input_size_info = OrderedDict()
blob_info['alexnet'] = _create_blob_info_dict(
    [('conv1', 4, 11, 0),
     ('norm1', 1, 1, 0),
     ('pool1', 2, 3, 0),
     ('conv2', 1, 5, 2),
     ('norm2', 1, 1, 0),
     ('pool2', 2, 3, 0),
     ('conv3', 1, 3, 1),
     ('conv4', 1, 3, 1),
     ('conv5', 1, 3, 1),
     ('pool5', 2, 3, 0)]
)
input_size_info['alexnet'] = (227, 227)

blob_info['caffenet'] = _create_blob_info_dict(
    [('conv1', 4, 11, 0),
     ('pool1', 2, 3, 0),
     ('norm1', 1, 1, 0),
     ('conv2', 1, 5, 2),
     ('pool2', 2, 3, 0),
     ('norm2', 1, 1, 0),
     ('conv3', 1, 3, 1),
     ('conv4', 1, 3, 1),
     ('conv5', 1, 3, 1),
     ('pool5', 2, 3, 0)]
)
input_size_info['caffenet'] = (227, 227)

blob_info['vgg16'] = _create_blob_info_dict(
    [('conv1_1', 1, 3, 1),
     ('conv1_2', 1, 3, 1),
     ('pool1', 2, 2, 0),
     ('conv2_1', 1, 3, 1),
     ('conv2_2', 1, 3, 1),
     ('pool2', 2, 2, 0),
     ('conv3_1', 1, 3, 1),
     ('conv3_2', 1, 3, 1),
     ('conv3_3', 1, 3, 1),
     ('pool3', 2, 2, 0),
     ('conv4_1', 1, 3, 1),
     ('conv4_2', 1, 3, 1),
     ('conv4_3', 1, 3, 1),
     ('pool4', 2, 2, 0),
     ('conv5_1', 1, 3, 1),
     ('conv5_2', 1, 3, 1),
     ('conv5_3', 1, 3, 1),
     ('pool5', 2, 2, 0)],
)
input_size_info['vgg16'] = (224, 224)

blob_info['vgg19'] = _create_blob_info_dict(
    [('conv1_1', 1, 3, 1),
     ('conv1_2', 1, 3, 1),
     ('pool1', 2, 2, 0),
     ('conv2_1', 1, 3, 1),
     ('conv2_2', 1, 3, 1),
     ('pool2', 2, 2, 0),
     ('conv3_1', 1, 3, 1),
     ('conv3_2', 1, 3, 1),
     ('conv3_3', 1, 3, 1),
     ('conv3_4', 1, 3, 1),
     ('pool3', 2, 2, 0),
     ('conv4_1', 1, 3, 1),
     ('conv4_2', 1, 3, 1),
     ('conv4_3', 1, 3, 1),
     ('conv4_4', 1, 3, 1),
     ('pool4', 2, 2, 0),
     ('conv5_1', 1, 3, 1),
     ('conv5_2', 1, 3, 1),
     ('conv5_3', 1, 3, 1),
     ('conv5_4', 1, 3, 1),
     ('pool5', 2, 2, 0)],
)
input_size_info['vgg19'] = (224, 224)
