#!/usr/bin/env bash

set -e
set -u


overwrite_plots=${1:-false}  # either "false", "overwrite", or a grep filter to pick specific logs by

# cd into repo root (parent directory)
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

function run_parselog_py() {
  overwrite_plots=$1
  fp_in=$2
  out_dir="data/_analysis/${fp_in#data/log/}"
  if [ "${overwrite_plots:-false}" = false -a -e "$out_dir" ] ; then
    echo skipping $fp_in
    exit
  fi
  out="$(mktemp -p ./data/tmp/)"
  out2="$(mktemp -p ./data/tmp/)"
  exec 3>"$out"
  exec 4<"$out"
  rm "$out"
  exec 5>"$out2"
  exec 6<"$out2"
  rm "$out2"
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  CYAN='\033[0;34m'
  NC='\033[0m' # No Color
  cmd="python ./bin/parselog.py ""$out_dir"" ""$fp_in"
  echo -e "$CYAN $cmd $NC"
  $cmd 1>&3 2>&5
  if [ $? -ne 0 ] ; then
    echo -e "$RED failed_to_parse $NC $fp_in"
    echo -e "$YELLOW "
    cat <&4  # stdout
    cat <&6  # stderr
  else
    echo -e "$GREEN successfully_parsed $NC $fp_in"
    grep Traceback $fp_in >/dev/null && echo -e "$YELLOW    WARN: but log contains a Traceback $NC"
    echo "    output_dir $out_dir"
    cat <&4  # stdout
    cat <&6  # stderr
  fi
}



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
