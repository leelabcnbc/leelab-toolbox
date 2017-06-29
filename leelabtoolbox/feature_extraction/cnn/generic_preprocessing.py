from __future__ import division, absolute_import, print_function, unicode_literals

from . import cnn_pkg_spec
import numpy as np
from pkgutil import get_data
from tempfile import TemporaryFile

# load the stat ImageNet image (as distributed with Caffe) for subtraction
with TemporaryFile() as f_temp:
    f_temp.write(get_data(cnn_pkg_spec, 'stat/ilsvrc_2012_mean.npy'))
    f_temp.seek(0)
    caffe_mu_image_bgr = np.load(f_temp)

caffe_mu_bgr = caffe_mu_image_bgr.mean(1).mean(1)  # this one is BGR

# based on <https://gist.github.com/ksimonyan/211839e770f7b538e2d8>
# the caffe model zoo about ILSVRC-2014 model (VGG team) with 16 weight layers
# this is bgr as well
vgg_mu_bgr = np.array([103.939, 116.779, 123.68])
