"""this is a rewrite of my cnn feature extraction routines
the old ones are at <https://github.com/leelabcnbc/early-vision-toolbox/tree/master/early_vision_toolbox/cnn>
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os.path
from collections import defaultdict

caffe_root = None
dir_dict = defaultdict(lambda: '.')  # by default, current directory.
# don't use __package__, which can be None. It's only for internal use, and we just don't care about it.
cnn_pkg_spec = __name__

try:
    import caffe

    caffe_root = os.path.normpath(caffe.__path__[0])
except ImportError:
    pass

if caffe_root is not None:
    # actually, these variables only hold if pycaffe is located in the source directory.
    # if you use Caffe from conda-forge, then you don't have these folders.
    dir_dict['caffe_models'] = os.path.normpath(os.path.join(caffe_root, '..', '..', 'models'))
    dir_dict['caffe_repo_root'] = os.path.normpath(os.path.join(caffe_root, '..', '..'))
    dir_dict['caffe_root'] = caffe_root
