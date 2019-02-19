#!/usr/bin/env bash

set -e
set -u

# cd into repo root (parent directory)
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

# define parselog function
function run_parselog_py() {
  fp_in=$1
  out_dir="data/_analysis/${fp_in#data/log/}"
  if [ -e "$out_dir" ] ; then exit ; fi
  python ./bin/parselog.py "$out_dir" "$fp_in" >/dev/null
}

# execute parselog in parallel
export -f run_parselog_py
find data/log/  -type f -name "*.log" -o -name "*.txt" \
    | parallel run_parselog_py
