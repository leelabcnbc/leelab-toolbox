#!/usr/bin/env bash

# this will only work on the CNBC cluster.
# leelab-toolbox-27 is currently just a cafferc3 env with name leelab-toolbox-27
. activate leelab-toolbox-27
. ~/DevOps/env_scripts/add_cudnn_v5.sh
. ~/DevOps/env_scripts/add_cuda_lib.sh
. ~/DevOps/env_scripts/add_conda_env_lib.sh
. ~/DevOps/env_scripts/add_openblas.sh
. ~/DevOps/env_scripts/add_caffe_latest_python.sh
