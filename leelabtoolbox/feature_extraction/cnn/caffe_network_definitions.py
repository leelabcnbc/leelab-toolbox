"""definitions of several off-the-shelf caffe networks"""

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import os.path
from . import dir_dict, cnn_pkg_spec
from pkgutil import get_data

_caffe_models_root = dir_dict['caffe_models']

net_info_dict = {}


def get_sub_prototxt_bytes(proto_struct, last_layer=None):
    result, net_base = proto_struct

    if last_layer is None:
        last_layer = list(result.keys())[-1]
    assert last_layer in result, "output layer {} not defined!".format(last_layer)

    prototxt_list = [net_base]
    done = False
    for layer in result:
        if not done:
            prototxt_list.append(result[layer])
            done = (layer == last_layer)
        else:
            break
    prototxt = b''.join(prototxt_list)
    return prototxt


def get_prototxt_bytes(filename):
    """notice that this returns bytes, which is str in Python 2"""
    return get_data(cnn_pkg_spec, 'prototxt/' + filename)


def _caffe_deploy_proto_by_layers(prototxt_bytes, num_layer_eat_dict, sep, base_length=1):
    assert isinstance(num_layer_eat_dict, OrderedDict)
    prototxt_text = prototxt_bytes.decode('utf-8')
    split_result = prototxt_text.split(sep)
    net_base = sep.join(split_result[:base_length]).encode('utf-8')
    split_result = split_result[base_length:]

    result = OrderedDict()
    for layer, layer_to_eat in num_layer_eat_dict.items():
        assert len(split_result) >= layer_to_eat
        result_this = sep + sep.join(split_result[:layer_to_eat])
        # we we still give back
        result[layer] = result_this.encode('utf-8')
        split_result = split_result[layer_to_eat:]
    assert not split_result  # at this time, it should be exactly empty.
    return result, net_base


def register_net_info(name, prototxt_path, num_layer_by_blob_dict, caffemodel_path, input_size, input_blob,
                      conv_blob_info_dict=None, sep=None):
    if sep is None:
        sep = 'layer {'

    # do value check
    assert isinstance(num_layer_by_blob_dict, OrderedDict)
    assert isinstance(conv_blob_info_dict, OrderedDict)
    assert isinstance(input_size, tuple) and len(input_size) == 2

    net_info_dict[name] = {
        'prototxt_path': prototxt_path,
        # this is for splitting prototxt. and can contain some uninteresting blobs, such as "data".
        'num_layer_by_blob_dict': num_layer_by_blob_dict,
        'caffemodel_path': caffemodel_path,
        'input_size': input_size,
        'input_blob': input_blob,
        'sep': sep,
        # all other information about the nearest conv layer under each blob,
        # here we only care about blobs obtained from convolution, and has no "data".
        'conv_blob_info_dict': conv_blob_info_dict

    }


def _create_blob_info_dict(info_dict_raw):
    info_dict = OrderedDict()
    for layer, stride, kernelsize, pad in info_dict_raw:
        info_dict[layer] = {
            'stride': stride,
            'kernelsize': kernelsize,
            'pad': pad
        }
    return info_dict


register_net_info('alexnet', prototxt_path='alexnet_deploy.prototxt',
                  num_layer_by_blob_dict=OrderedDict(
                      [('data', 1),
                       ('conv1', 2),
                       ('norm1', 1),
                       ('pool1', 1),
                       ('conv2', 2),
                       ('norm2', 1),
                       ('pool2', 1),
                       ('conv3', 2),
                       ('conv4', 2),
                       ('conv5', 2),
                       ('pool5', 1),
                       ('fc6', 3),
                       ('fc7', 3),
                       ('fc8', 1),
                       ('prob', 1)]
                  ), caffemodel_path=os.path.join(_caffe_models_root, 'bvlc_alexnet', 'bvlc_alexnet.caffemodel'),
                  input_size=(227, 227),
                  input_blob='data',
                  conv_blob_info_dict=_create_blob_info_dict(
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
                  ))

register_net_info('caffenet', prototxt_path='caffenet_deploy.prototxt',
                  num_layer_by_blob_dict=OrderedDict(
                      [('data', 1),
                       ('conv1', 2),
                       ('pool1', 1),
                       ('norm1', 1),
                       ('conv2', 2),
                       ('pool2', 1),
                       ('norm2', 1),
                       ('conv3', 2),
                       ('conv4', 2),
                       ('conv5', 2),
                       ('pool5', 1),
                       ('fc6', 3),
                       ('fc7', 3),
                       ('fc8', 1),
                       ('prob', 1)]
                  ), caffemodel_path=os.path.join(_caffe_models_root, 'bvlc_reference_caffenet',
                                                  'bvlc_reference_caffenet.caffemodel'), input_size=(227, 227),
                  input_blob='data',
                  conv_blob_info_dict=_create_blob_info_dict(
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
                  ))

register_net_info('vgg16', prototxt_path='VGG_ILSVRC_16_layers_deploy.prototxt',
                  num_layer_by_blob_dict=OrderedDict(
                      [('conv1_1', 2),
                       ('conv1_2', 2),
                       ('pool1', 1),
                       ('conv2_1', 2),
                       ('conv2_2', 2),
                       ('pool2', 1),
                       ('conv3_1', 2),
                       ('conv3_2', 2),
                       ('conv3_3', 2),
                       ('pool3', 1),
                       ('conv4_1', 2),
                       ('conv4_2', 2),
                       ('conv4_3', 2),
                       ('pool4', 1),
                       ('conv5_1', 2),
                       ('conv5_2', 2),
                       ('conv5_3', 2),
                       ('pool5', 1),
                       ('fc6', 3),
                       ('fc7', 3),
                       ('fc8', 1),
                       ('prob', 1)],
                  ), caffemodel_path=os.path.join(_caffe_models_root, '211839e770f7b538e2d8',
                                                  'VGG_ILSVRC_16_layers.caffemodel'), input_size=(224, 224),
                  input_blob='data',
                  conv_blob_info_dict=_create_blob_info_dict(
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
                  ),
                  sep='layers {')

register_net_info('vgg19', prototxt_path='VGG_ILSVRC_19_layers_deploy.prototxt',
                  num_layer_by_blob_dict=OrderedDict(
                      [('conv1_1', 2),
                       ('conv1_2', 2),
                       ('pool1', 1),
                       ('conv2_1', 2),
                       ('conv2_2', 2),
                       ('pool2', 1),
                       ('conv3_1', 2),
                       ('conv3_2', 2),
                       ('conv3_3', 2),
                       ('conv3_4', 2),
                       ('pool3', 1),
                       ('conv4_1', 2),
                       ('conv4_2', 2),
                       ('conv4_3', 2),
                       ('conv4_4', 2),
                       ('pool4', 1),
                       ('conv5_1', 2),
                       ('conv5_2', 2),
                       ('conv5_3', 2),
                       ('conv5_4', 2),
                       ('pool5', 1),
                       ('fc6', 3),
                       ('fc7', 3),
                       ('fc8', 1),
                       ('prob', 1)],
                  ), caffemodel_path=os.path.join(_caffe_models_root, '3785162f95cd2d5fee77',
                                                  'VGG_ILSVRC_19_layers.caffemodel'), input_size=(224, 224),
                  input_blob='data',
                  conv_blob_info_dict=_create_blob_info_dict(
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
                  ),
                  sep='layers {')

proto_struct_dict = {key: _caffe_deploy_proto_by_layers(get_prototxt_bytes(value['prototxt_path']),
                                                        value['num_layer_by_blob_dict'],
                                                        sep=value['sep']) for key, value in
                     net_info_dict.items()}
