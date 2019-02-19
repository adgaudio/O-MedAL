#!/bin/bash
#SBATCH -N 1
#SBATCH -n {{cpu_cores}}
#SBATCH -p {{partition}}
#SBATCH --gres={{gres}}
#SBATCH -t {{maxtime}}

# Helper function to ensure only one instance of a job runs at a time.
# Optionally, on finish, can write a file to ensure the job won't run again.
# usage: use_lockfile myfile.locked [ myfile.finished ]
function use_lockfile() {
  lockfile_fp="${1}.running"
  lockfile_runonce="$2"
  lockfile_success_fp="${1}.finished"
  # create lock file
  if [ -e "$lockfile_fp" ] ; then
    echo "job already running!"
    exit
  fi
  if [ "$lockfile_runonce" = "yes" -a -e "$lockfile_success_fp" ] ; then
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
    cd "$(dirname "$(realpath "$lockfile_fp")")"
    rm "$(basename "$lockfile_fp")"
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


# load environment
export TERM=screen
module load AI/anaconda3-5.1.0_gpu.2018-08
source activate $AI_ENV
cd $SCRATCH/medal_improvements
source ./data/.bridges_venv/bin/activate

pwd

set -e
set -u

# choose the name of this lockfile (ie "myjob.running") to guarantee only one
# of this job will run at a time.  the second parameter (ie "myjob.finished")
# is optional, and will guarantee the job won't run again.
use_lockfile {{lockfile_path}} {{lockfile_runonce}}

echo "Running MedAL on hostname: $(hostname)"
echo "run_id: {{run_id}}"
date

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

python -m medal {{python_args}} --run-id {{run_id}}

echo sbatch job finished
date
