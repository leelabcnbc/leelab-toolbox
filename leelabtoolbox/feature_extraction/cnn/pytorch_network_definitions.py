"""definitions of several off-the-shelf networks in pytorch"""

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict

# here, I only need a correspondence between PyTorch modules and caffe blobs.
blob_corresponding_info = dict()
blob_corresponding_info['alexnet'] = OrderedDict([('conv1', 'features.1'),
                                                  ('norm1', 'features.2'),
                                                  ('pool1', 'features.3'),
                                                  ('conv2', 'features.5'),
                                                  ('norm2', 'features.6'),
                                                  ('pool2', 'features.7'),
                                                  ('conv3', 'features.9'),
                                                  ('conv4', 'features.11'),
                                                  ('conv5', 'features.13'),
                                                  ('pool5', 'features.14'),
                                                  ('fc6', 'classifier.2'),
                                                  ('fc7', 'classifier.5')])

blob_corresponding_info['caffenet'] = OrderedDict([('conv1', 'features.1'),
                                                   ('pool1', 'features.2'),
                                                   ('norm1', 'features.3'),
                                                   ('conv2', 'features.5'),
                                                   ('pool2', 'features.6'),
                                                   ('norm2', 'features.7'),
                                                   ('conv3', 'features.9'),
                                                   ('conv4', 'features.11'),
                                                   ('conv5', 'features.13'),
                                                   ('pool5', 'features.14'),
                                                   ('fc6', 'classifier.2'),
                                                   ('fc7', 'classifier.5')])

blob_corresponding_info['vgg16'] = OrderedDict([('conv1_1', 'features.1'),
                                                ('conv1_2', 'features.3'),
                                                ('pool1', 'features.4'),
                                                ('conv2_1', 'features.6'),
                                                ('conv2_2', 'features.8'),
                                                ('pool2', 'features.9'),
                                                ('conv3_1', 'features.11'),
                                                ('conv3_2', 'features.13'),
                                                ('conv3_3', 'features.15'),
                                                ('pool3', 'features.16'),
                                                ('conv4_1', 'features.18'),
                                                ('conv4_2', 'features.20'),
                                                ('conv4_3', 'features.22'),
                                                ('pool4', 'features.23'),
                                                ('conv5_1', 'features.25'),
                                                ('conv5_2', 'features.27'),
                                                ('conv5_3', 'features.29'),
                                                ('pool5', 'features.30'),
                                                ('fc6', 'classifier.1'),
                                                ('fc7', 'classifier.4')])

blob_corresponding_info['vgg19'] = OrderedDict([('conv1_1', 'features.1'),
                                                ('conv1_2', 'features.3'),
                                                ('pool1', 'features.4'),
                                                ('conv2_1', 'features.6'),
                                                ('conv2_2', 'features.8'),
                                                ('pool2', 'features.9'),
                                                ('conv3_1', 'features.11'),
                                                ('conv3_2', 'features.13'),
                                                ('conv3_3', 'features.15'),
                                                ('conv3_4', 'features.17'),
                                                ('pool3', 'features.18'),
                                                ('conv4_1', 'features.20'),
                                                ('conv4_2', 'features.22'),
                                                ('conv4_3', 'features.24'),
                                                ('conv4_4', 'features.26'),
                                                ('pool4', 'features.27'),
                                                ('conv5_1', 'features.29'),
                                                ('conv5_2', 'features.31'),
                                                ('conv5_3', 'features.33'),
                                                ('conv5_4', 'features.35'),
                                                ('pool5', 'features.36'),
                                                ('fc6', 'classifier.1'),
                                                ('fc7', 'classifier.4')])

blob_corresponding_reverse_info = dict()
for net_name, net_info in blob_corresponding_info.items():
    blob_corresponding_reverse_info[net_name] = OrderedDict()
    for x, y in net_info.items():
        blob_corresponding_reverse_info[net_name][y] = x
    assert len(blob_corresponding_reverse_info[net_name]) == len(net_info)
