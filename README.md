# rllab

## Setup Instructions

- Install pip libraries:

  ```
  pip install joblib pyprind
  ```

- Install pygame:

  If using Anaconda, run the following:

  ```
  # linux
  conda install -c https://conda.binstar.org/tlatorre pygame
  # mac
  conda install -c https://conda.anaconda.org/quasiben pygame
  ```

  Otherwise, follow the official instructions.

- Install box2d:

  If using Anaconda, run the following:

  ```
  conda install -c https://conda.anaconda.org/kne pybox2d
  ```

  Otherwise, follow the official instructions.

## Running Experiments

  The following bash command runs the Proximal Policy Optimization algorithm on cartpole using 4 sampling processes in parallel.

  ```
  python scripts/run_experiment.py \
        --algo ppo \
        --mdp box2d.cartpole_mdp \
        --normalize_mdp \
        --policy mean_std_nn_policy \
        --vf mujoco_value_function \
        --exp_name vpg_box2d_cartpole \
        --n_parallel 4 \
        --snapshot_mode all \
        --algo_binary_search_penalty False \
        --algo_whole_paths True \
        --algo_batch_size 50000 \
        --algo_max_path_length 100 \
        --algo_n_itr 500 \
        --seed 1
  ```

  See [Recipies](https://github.com/dementrock/rllab/wiki/Recipies) for more.

## Visualize a policy

  While running an experiment, all the relevant parameters will be stored in a pickle file under data/EXP_NAME. This file can be used to visualize the policy using the following command:

  ```
  python scripts/sim_policy.py PKL_FILE
  ```
