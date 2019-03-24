#!/usr/bin/env bash

set -e
set -u

overwrite_plots=${1:-false}  # either "false", "overwrite", or a grep filter to pick specific logs by

# cd into repo root (parent directory)
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

source ./bin/bash_lib.sh

export -f run_parselog_py

if [ "$overwrite_plots" = "false" ] ; then
  # execute parselog in parallel on files we don't have plots for
  comm -23 \
    <( find data/log/  -type f -name "*.log" -o -type f -name "*.txt" |sort) \
    <( find data/_analysis/ -type d | sed 's/_analysis/log/' |sort ) | \
    parallel run_parselog_py $overwrite_plots
elif [ "$overwrite_plots" = "overwrite" ] ; then
  # execute parselog in parallel on all log files found
  find data/log/  -type f -name "*.log" -o -type f -name "*.txt" \
      | parallel run_parselog_py $overwrite_plots
else
  # execute parselog in parallel on all log files found
  find data/log/  -type f -name "*.log" -o -type f -name "*.txt" \
      | grep $overwrite_plots \
      | parallel run_parselog_py "$overwrite_plots"
fi
