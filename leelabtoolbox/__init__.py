import os.path
_leelabtoolbox_root = os.path.split(__file__)[0]
assert os.path.isabs(_leelabtoolbox_root)
print(_leelabtoolbox_root)