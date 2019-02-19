#!/usr/bin/env bash

set -e
set -u

bridges_user=${1}

rsync -ave ssh $bridges_user@data.bridges.psc.edu:/pylon5/ci4s8dp/$bridges_user/medal_improvements/data/log ./data/
