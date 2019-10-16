#!/usr/bin/env bash
set -e
set -u

cd "$(dirname "$(dirname "$(realpath "$0")")")"
pwd

. ./bin/bash_lib.sh
export -f run_cmd_and_log
export -f run_cmd
export -f log_initial_msgs
export -f use_lockfile

export device="cuda:1"

# # run_id=R3 python_args="MedalResnet18BinaryClassifier --learning-rate 0.003" maxtime=5:00:00 partition=GPU-shared ./bridges/rsync_to_bridges_and_run.sh agaudio sbatch

# ri=RM6g ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 20 --device cuda --checkpoint-fname '{config.run_id}/al_{config.cur_al_iter}.pth'"
# ri=RM6e ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 10 --device cuda --checkpoint-fname '{config.run_id}/al_{config.cur_al_iter}.pth'"

# ri=RMO6-37.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.375"
# ri=RMO6-62.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.625"
# (ri=RMO6-12.5e ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.125 --device cuda:0") &
# (ri=RMO6-87.5e ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.875 --device cuda:1") &
wait
# ri=RMO6-0d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0"
# ri=RMO6-50d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.5 "
# ri=RMO6-100d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 1 "
# ri=RMO6-25d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.25 "
# ri=RMO6-75d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.75 "


# get wall time on most useful omedal results
(ri=RMO6-12.5e ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.125 --device cuda:0") &
(ri=RMO6-87.5e ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.875 --device cuda:1") &
wait
# wall time for baseline for 80 epochs.
(ri=R6b ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 80 --device cuda:0") &
# get wall time AND performance for omedal with 20 epochs per al iter
(ri=RMO6-87.5-20epocha ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.875 --device cuda:1 --epochs 20") &
wait
# and get wall time estimates for medal.
# wait
(ri=RM6h ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 20 --device cuda:1") &
(ri=RM6i ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 10 --device cuda:0") &
wait
