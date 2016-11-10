#!/usr/bin/env bash

# this will only work on the CNBC cluster.
# leelab-toolbox-27 is currently just a cafferc3 env with name leelab-toolbox-27
. activate leelab-toolbox-27
. ~/DevOps/env_scripts/add_cudnn_v5.sh
. ~/DevOps/env_scripts/add_cuda_lib.sh
. ~/DevOps/env_scripts/add_conda_env_lib.sh
. ~/DevOps/env_scripts/add_openblas.sh
. ~/DevOps/env_scripts/add_caffe_latest_python.sh

# add leelabtoolbox to PYTHONPATH

# from <http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in?page=1&tab=votes#tab-top>
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${DIR}":"${PYTHONPATH}"
