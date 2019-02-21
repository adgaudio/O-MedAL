#!/usr/bin/env bash

# Run MedAL on Bridges (Pittsburgh Super Computer) by tunneling through a jump
# server.

function usage() {
  cat <<EOF
usage: 
There are 3 ways to use this script.

The most common and useful way is to submit a batch job.  This is set up to only run an experiment at most once by default.

run_id=my_test_experiment python_args="baseline-inception" maxtime=3:00:00 ./bridges/rsync_to_bridges_and_run.sh agaudio sbatch

You can also set up the interactive mode (which also saves a log file of the
run and logs you into an interactive session).  To use this interactive mode,
you should add the following to your ssh config.

Host jump
  Hostname IP_ADDRESS_OF_YOUR_JUMP_HOST
  User YOUR_USERNAME

Then the interactive command looks like:

    run_id=test python_args="baseline-inception" ./bridges/rsync_to_bridges_and_run.sh agaudio interactive

Third, you can also create your own sbatch file and execute it.  The log
files will automatically get saved in the log directory.

    run_id=test ./bridges/rsync_to_bridges_and_run.sh agaudio ./path/to/myfile.sbatch
EOF
  exit 1
}

set -u
set -e
# set -x

if [ "${1:-}" = "" ] ; then usage ; fi
bridges_user=${1}
mode=${2:-donothing}  # interactive|sbatch|some/filepath.sbatch  # if interactive, limited to 8 hours.


# cd into parent directory of the script is
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
pwd

# set a directory for this run
bdir="/pylon5/ci4s8dp/$bridges_user/"
run_dir="$bdir/medal_runs/$run_id"

curtime="$(date +%Y%m%dT%H%M%S)"
log_fp="$run_dir/data/log/$run_id-$curtime.log"

# rsync latest code over and set up data directory
if [ "${bridges_user}" != "no" ] ; then
ssh $bridges_user@bridges.psc.edu <<EOF
mkdir -p "$run_dir"
ln -rs "$bdir/medal_improvements/data" "$run_dir" || true
EOF
rsync -ave ssh --delete --exclude __pycache__ --exclude data --exclude old \
  ./ "$bridges_user@data.bridges.psc.edu:$bdir/medal_runs/$run_id"
fi

if [ "${mode}" = "interactive" ] ; then
# # set up to run interactively on bridges (via a socks proxy running tmux)
# # within bridges, load the gpu interactively and run code inside a tmux session
# # (so can consult nvidia-smi or do other things on the machine while it runs)
TERM=screen ssh -tt -A jump 'tmux new-session -A -s 0 \; new-window -t 0:. -n MedAL -a ssh  '"$bridges_user"'@bridges.psc.edu' <<EOF
interact -p ${partition:-GPU-small} --gres=${gres:-gpu:p100:2} -t ${maxtime:-08:00:00} -N 1 -n 28

module load AI/anaconda3-5.1.0_gpu.2018-08 
source activate \$AI_ENV
cd $run_dir
source ./data/.bridges_venv/bin/activate
pwd

# rm Model_save.hdf5
# tmux new-session "python Script.py 2>&1 | tee -a data/log/`date +%Y%m%dT%H%M%S`.log"
tmux new-session "python -m medal $python_args --run-id $run_id 2>&1 | tee -a $log_fp"

exit
EOF

elif [ "$mode" = "sbatch" ] ; then
  mkdir -p data/tmp/sbatch

  sbatch_fp="data/tmp/sbatch/$run_id-$curtime.sbatch"


  # All these options are settable from the command-line
  # some are required
  run_id=${run_id} \
  run_dir=${run_dir} \
  python_args=${python_args} \
  cpu_cores=${cpu_cores:-28} \
  partition=${partition:-GPU-shared} \
  gres=${gres:-gpu:p100:2} \
  maxtime=${maxtime:-15:00:00} \
  lockfile_runonce=${lockfile_runonce:-yes} \
  lockfile_path=data/run/$run_id \
  j2 -f env \
  bridges/template.sbatch |tee "$sbatch_fp"

  echo Wrote tmp sbatch to: $sbatch_fp

  rsync -aqe ssh ./data/tmp/sbatch \
    $bridges_user@data.bridges.psc.edu:$run_dir/data/tmp

  ssh $bridges_user@bridges.psc.edu <<EOF
cd $run_dir
sbatch -o "$log_fp" -e "$log_fp" $sbatch_fp
EOF

elif [ -e "$mode" ] ; then # run via sbatch
  echo Sbatch file should exist locally and remotely. Running it:
  echo $mode

ssh $bridges_user@bridges.psc.edu <<EOF
sbatch -o $log_fp -e $log_fp $mode
EOF

else
  echo Do nothing further
fi

