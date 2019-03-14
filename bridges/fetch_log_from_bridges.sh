#!/usr/bin/env bash

set -e
set -u

bridges_user=${1}

# cd into parent directory of the script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# copy data
fps="$(rsync -ae ssh -k --out-format="%f\n" $bridges_user@data.bridges.psc.edu:/pylon5/ci4s8dp/$bridges_user/medal_improvements/data/log ./data/)"
echo fetched:
echo -e $fps

# generate plots of copied data
. ./bin/bash_lib.sh
export -f run_parselog_py
echo -e $fps |  grep / | awk '{$1=$1};1' | parallel run_parselog_py overwrite data/{}
