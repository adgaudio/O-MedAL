# a library of helper functions for scripting
# shell scripts should source this library.


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
      if [ "$lockfile_runonce" = "yes" ] ; then
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


function log_initial_msgs() {(
  set -e
  set -u
  run_id=$1
  echo "Running on hostname: $(hostname)"
  echo "run_id: ${run_id}"
  date

  # print out current configuration
  echo ======================
  echo CURRENT GIT CONFIGURATION:
  echo "git commit: $(git rev-parse HEAD)"
  echo
  echo git status:
  git status
  echo
  echo git diff:
  git --no-pager diff --cached
  git --no-pager diff
  echo
  echo ======================
  echo
  echo
)}

function run_cmd() {
  run_id="$1"
  cmd="$2"
  export -f log_initial_msgs
cat <<EOF | sh
log_initial_msgs $run_id
echo "$cmd"
$cmd
echo job finished
date
EOF
}
function run_cmd_and_log() {
  run_id="$1"
  cmd="$2"
  lockfile_path=./data/run/$run_id
  lockfile_runonce=yes

  (
  use_lockfile ${lockfile_path} ${lockfile_runonce}
  log_fp="./data/log/$run_id-`date +%Y%m%dT%H%M%S`.log"
  run_cmd "$run_id" "$cmd" 2>&1 | tee $log_fp
  )
}
