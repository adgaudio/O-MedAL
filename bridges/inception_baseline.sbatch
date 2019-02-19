#!/bin/bash
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:p100:2
#SBATCH -t 15:00:00

# interact -p GPU-small --gres=gpu:p100:2 -t 08:00:00 -N 1 -n 28

set -e
set -u


# Helper function to ensure only one instance of a job runs at a time.
# Optionally, on finish, can write a file to ensure the job won't run again.
# usage: use_lockfile myfile.locked [ myfile.finished ]
function use_lockfile() {
  lockfile_fp="$(realpath ${1:-./lockfile})"
  lockfile_success_fp=${2:-$lockfile_fp}
  # create lock file
  if [ -e "$lockfile_fp" ] ; then
    echo "job already running!"
    exit
  fi
  if [ -e "$lockfile_success_fp" ] ; then
    echo "job previously completed!"
    exit
  fi
  runid=$RANDOM
  echo $runid > "$lockfile_fp"

  # check that there wasn't a race condition
  # (not guaranteed to work but should be pretty good)
  sleep $(bc -l <<< "scale=4 ; ${RANDOM}/32767/10")
  rc=0
  grep $runid "$lockfile_fp" || rc=1
  if [ "$rc" = "1" ] ; then
    echo caught race condition 
    exit 1
  fi

  # automatically remove the lockfile when finished, whether fail or success
  function remove_lockfile() {
    # echo removed lockfile $lockfile_fp
    cd "$(dirname $lockfile_fp)"
    rm "$lockfile_fp"
  }
  function trap_success() {
    if [ "$?" = "0" ] ; then
      echo job successfully completed
      if [ "$lockfile_fp" != "${lockfile_success_fp}" ] ; then
        echo please rm this file to re-run job again: ${lockfile_success_fp}
        date > $lockfile_success_fp
      fi
    fi
    remove_lockfile
  }
  function trap_err() {
    echo "ERROR on line $(caller)" >&2
    exit 1
  }
  trap trap_err ERR
  trap trap_success 0
  # trap remove_lockfile EXIT
}


# choose the name of this lockfile (ie "myjob.running") to guarantee only one
# of this job will run at a time.  the second parameter (ie "myjob.finished")
# is optional, and will guarantee the job won't run again.
use_lockfile inception_baseline.running


echo "Running MedAL on hostname: $(hostname)"
date

# load environment
export TERM=screen

cd $SCRATCH/medal_improvements
source ./data/.bridges_venv/bin/activate
pwd

# print out current configuration
echo ======================
echo CURRENT GIT CONFIGURATION:
echo "git commit: $(git rev-parse HEAD)"
echo
echo git status:
git status
echo
echo ======================
echo
echo

python -m medal

echo finished training MedAL
date
