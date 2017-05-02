# Project setup script
# Source this file to set up the environment for this project.

export CUDA_VISIBLE_DEVICES=''

export RLLAB_HOME="$(pwd)"

alias ph="cd $RLLAB_HOME"

export PYTHONPATH="$RLLAB_HOME:$PYTHONPATH"


alias set_display="export DISPLAY=':0.0'"
alias unset_display="unset DISPLAY"

alias viskit="python $RLLAB_HOME/rllab/viskit/frontend.py"

alias ec2scp="scp -i $RLLAB_HOME/private/key_pairs/rllab-us-west-1.pem"

function ec2ssh {
  ssh -o "UserKnownHostsFile /dev/null" -o "StrictHostKeyChecking=no" -i "$RLLAB_HOME"'/private/key_pairs/rllab-us-west-1.pem' 'ubuntu@'"$1"
}

alias s3sync="python $RLLAB_HOME/scripts/sync_s3.py"

function sim_policy {
    (
      export PYTHONPATH="$(pwd):$PYTHONPATH"
      python "$RLLAB_HOME/scripts/sim_policy.py" "$@"
    )
}

export MPLBACKEND='Agg'

source activate rllab_goal_rl
