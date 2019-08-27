#!/usr/bin/env bash
set -e
set -u

cd "$(dirname "$(realpath "$0")")"
pwd

. ./bin/bash_lib.sh
export -f run_cmd_and_log
export -f run_cmd
export -f log_initial_msgs
export -f use_lockfile


# # run_id=R2 python_args="MedalResnet18BinaryClassifier --learning-rate 0.002" maxtime=5:00:00 partition=GPU-shared ./bridges/rsync_to_bridges_and_run.sh agaudio sbatch
# # run_id=R3 python_args="MedalResnet18BinaryClassifier --learning-rate 0.003" maxtime=5:00:00 partition=GPU-shared ./bridges/rsync_to_bridges_and_run.sh agaudio sbatch

# ri=RM6 ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150"
ri=RM6g ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 20 --device cuda --checkpoint-fname '{config.run_id}/al_{config.cur_al_iter}.pth'"

# TODO run_cmd_and_log RM6b "python -m medal MedalResnet18BinaryClassifier --learning-rate 0.003 --epochs 150"

# ri=RMO6-37.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.375"
# ri=RMO6-62.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.625"
# ri=RMO6-12.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.125"
# ri=RMO6-87.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.875"
# ri=RMO6-0d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0"
# ri=RMO6-50d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.5 "
# ri=RMO6-100d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 1 "
# ri=RMO6-25d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.25 "
# ri=RMO6-75d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.75 "
