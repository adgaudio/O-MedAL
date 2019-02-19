#!/usr/bin/env bash

set -e
set -u

# cd into repo root (parent directory)
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

source ./bin/bash_lib.sh

# execute parselog in parallel
export -f run_parselog_py
find data/log/  -type f -name "*.log" -o -type f -name "*.txt" \
    | parallel run_parselog_py
