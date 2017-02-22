# Project setup script
# Source this file to set up the environment for this project.

export RLLAB_HOME="$(pwd)"

alias ph="cd $RLLAB_HOME"

export PYTHONPATH="$RLLAB_HOME:$PYTHONPATH"


alias set_display="export DISPLAY=':0.0'"
alias unset_display="unset DISPLAY"

alias viskit="python $RLLAB_HOME/rllab/viskit/frontend.py"

function sim_policy {
    (
      export PYTHONPATH="$(pwd):$PYTHONPATH"
      python "$RLLAB_HOME/scripts/sim_policy.py" "$@"
    )
}

export MPLBACKEND='Agg'

source activate rllab_goal_rl
