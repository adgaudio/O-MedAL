#!/usr/bin/env bash

set -e
set -u

bridges_user=${1}
parse_all_logs=${2:-yes}

# cd into parent directory of the script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

rsync -ave ssh $bridges_user@data.bridges.psc.edu:/pylon5/ci4s8dp/$bridges_user/medal_improvements/data/log ./data/

if [ "$parse_all_logs" = "yes" ] ; then
  ./bin/parse_all_logs.sh
fi
