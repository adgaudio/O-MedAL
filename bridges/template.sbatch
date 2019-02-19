#!/bin/bash
#SBATCH -N 1
#SBATCH -n {{cpu_cores}}
#SBATCH -p {{partition}}
#SBATCH --gres={{gres}}
#SBATCH -t {{maxtime}}

# load environment
export TERM=screen
module load AI/anaconda3-5.1.0_gpu.2018-08
source activate $AI_ENV
cd {{run_dir}}
source ./data/.bridges_venv/bin/activate

pwd

set -e
set -u

# choose the name of this lockfile (ie "myjob.running") to guarantee only one
# of this job will run at a time.  the second parameter (ie "myjob.finished")
# is optional, and will guarantee the job won't run again.
source ./bin/bash_lib.sh
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
