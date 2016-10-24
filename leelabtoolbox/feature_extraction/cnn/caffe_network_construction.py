from __future__ import division, absolute_import, print_function, unicode_literals

from tempfile import NamedTemporaryFile
from os import remove

try:
    import caffe
except ImportError:
    pass

from .caffe_network_definitions import net_info_dict, get_prototxt_bytes


def create_empty_net(prototxt, model_weight_name=None):
    """ create an uninitialized caffe model.
    """

    # you must provide two parts as those predefined files

    # create a temporary file... fuck Caffe
    f_temp_file = NamedTemporaryFile(delete=False)
    f_temp_file.write(prototxt)
    f_temp_file.close()
    # create the net.
    if model_weight_name is None:
        created_net = caffe.Net(f_temp_file.name, caffe.TEST)
    else:
        created_net = caffe.Net(f_temp_file.name, model_weight_name, caffe.TEST)
    # remove the temporary file
    remove(f_temp_file.name)
    return created_net


def create_predefined_net(name, load_weight=True):
    """ create one example CNN in caffe examples.

    :param name:
    :return:
    """
    if load_weight:
        model_weight_name = net_info_dict[name]['caffemodel_path'].encode('utf-8')
    else:
        model_weight_name = None
    model_file_name = net_info_dict[name]['prototxt_path']
    prototxt = get_prototxt_bytes(model_file_name)
    return create_empty_net(prototxt, model_weight_name)


def fill_weights(src_net, dest_net):
    """fill up weight data from one net to another

    :param src_net:
    :param dest_net:
    :return:
    """

    # then fill up weights from src to dest.

    assert set(dest_net.params.keys()) <= set(src_net.params.keys()), "some layers non existent in src net!"

    # changing value in place is safe.
    for layer_name, param in dest_net.params.iteritems():
        assert len(param) == len(src_net.params[layer_name])
        for idx in range(len(param)):
            param[idx].data[...] = src_net.params[layer_name][idx].data
    return dest_net
