"""relevant functions to make matlab work."""
from __future__ import absolute_import, division, print_function, unicode_literals
import os.path
from . import _leelabtoolbox_root
import matlab.engine

_matlab_handle_dict = {
    'handle': None
}

vlfeat_root = os.path.join(_leelabtoolbox_root, '..', '3rdparty', 'vlfeat')


def get_matlab_handle():
    if _matlab_handle_dict['handle'] is None:
        _matlab_handle_dict['handle'] = matlab.engine.start_matlab()
        # also, add vlfeat stuff.
        _matlab_handle_dict['handle'].run(os.path.join(vlfeat_root, 'toolbox', 'vl_setup'), nargout=0)
    return _matlab_handle_dict['handle']
