#!/usr/bin/env bash

conda install --yes --no-update-dependencies -c conda-forge scikit-learn joblib scikit-image h5py
pip install imagen==2.0.1
# TODO: add spams if using Python 2.7

# TODO: install matlab plugin
# under matlabroot\extern\engines\python folder
# python setup.py build --build-base=/tmp install
# Don't use arbitrary folder such as $HOME! Somehow if you have `lib` under that folder, files under that lib will be
# installed as well!
# such as
# creating /home/yimengzh/miniconda2/envs/leelab-toolbox/lib/python3.4/site-packages/OpenBLAS
# creating /home/yimengzh/miniconda2/envs/leelab-toolbox/lib/python3.4/site-packages/OpenBLAS/info
# copying /home/yimengzh/lib/OpenBLAS/info/has_prefix -> /home/yimengzh/miniconda2/envs/leelab-toolbox/lib/python3.4/site-packages/OpenBLAS/info
# copying /home/yimengzh/lib/OpenBLAS/info/about.json -> /home/yimengzh/miniconda2/envs/leelab-toolbox/lib/python3.4/site-packages/OpenBLAS/info
# copying /home/yimengzh/lib/OpenBLAS/info/index.json -> /home/yimengzh/miniconda2/envs/leelab-toolbox/lib/python3.4/site-packages/OpenBLAS/info
#
# seems that this is by design of distutils, not by MATLAB.

# travis specific, handling caffe
if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
    # fix protobuf version, as cafferc3 here in conda-forge might be compiled with 3.0.0b3, and somehow it can't work
    # with newer version.
    conda install --yes --no-update-dependencies -c conda-forge caffe protobuf=3.1.0 gflags=2.1.2 opencv=3.1
fi
