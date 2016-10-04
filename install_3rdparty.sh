#!/usr/bin/env bash

# get dir of script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${DIR}/3rdparty"
mkdir vlfeat
tar -xzvf vlfeat-0.9.20-bin.tar.gz --strip-components=1 -C vlfeat
