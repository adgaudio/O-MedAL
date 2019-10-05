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


## Resnet18 baseline  (we actually used only 80 epochs in the paper plots)
# ri=R6 ; run_cmd_and_log $ri "python -m medal BaselineResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150"

## MedAL with patience
# ri=RM6 ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150"
# ri=RM6e ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 10 --device cuda --checkpoint-fname '{config.run_id}/al_{config.cur_al_iter}.pth'"
# ri=RM6g ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 20 --device cuda --checkpoint-fname '{config.run_id}/al_{config.cur_al_iter}.pth'"

## OMedAL with 10 epochs
# ri=RMO6-37.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.375"
# ri=RMO6-62.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.625"
# ri=RMO6-12.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.125"
# ri=RMO6-87.5d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.875"
# ri=RMO6-0d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0"
# ri=RMO6-50d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.5 "
# ri=RMO6-100d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 1 "
# ri=RMO6-25d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.25 "
# ri=RMO6-75d ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.75 "


# get wall time on most useful omedal results (the wall time column in fig. 5 only uses time up to the given al iteration for each row)
(ri=RMO6-12.5e ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.125 --device cuda:0") &
(ri=RMO6-87.5e ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 10 --online-sample-frac 0.875 --device cuda:1") &
wait
# wall time for baseline for 80 epochs.
(ri=R6b ; run_cmd_and_log $ri "python -m medal BaselineResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 80 --checkpoint-interval 0 --device cuda:0") &
wait
# and get wall time estimates for medal
(ri=RM6h ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 20 --device cuda:1") &
(ri=RM6i ; run_cmd_and_log $ri "python -m medal MedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 10 --device cuda:0") &
wait

# performance and wall time for omedal p=87.5 with 20 epochs or with patience (for better data labeling efficiency and accuracy results)
(ri=RMO6-87.5-5patience ;  run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 5 --online-sample-frac 0.875 --device cuda:0 ") &
(ri=RMO6-87.5-20epoch ;   run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 20  --online-sample-frac 0.875 --device cuda:1 ") &
wait
(ri=RMO6-87.5-10patience ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 10 --online-sample-frac 0.875 --device cuda:0 ") &
(ri=RMO6-87.5-20patience ; run_cmd_and_log $ri "python -m medal OnlineMedalResnet18BinaryClassifier --run-id $ri --learning-rate 0.003 --epochs 150 --early-stopping-patience 20 --online-sample-frac 0.875 --device cuda:0 ") &
wait

# TODO: 20 epoch, some patience
