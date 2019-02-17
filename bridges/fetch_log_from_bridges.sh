#!/usr/bin/env bash

set -e
set -u

bridges_user=${1:-no}
parse_log=${2:-no}

if [ "$bridges_user" != "no" ] ; then
  rsync -ave ssh $bridges_user@data.bridges.psc.edu:/pylon5/ci4s8dp/$bridges_user/medal_improvements/data/log ./data/
fi

cd "$(dirname $(readlink -f "$0"))"

if [ "${parse_log}" != "no" ] ; then
  fp="data/log/$(ls -tr ./data/log | tail -n 1)"
  img_dir="data/analysis/$(basename "$fp")"
  mkdir -p $img_dir
  python parselog.py $img_dir $fp
fi
