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

  The following bash command runs the Proximal Policy Optimization algorithm on cartpole.

  ```
python scripts/run_experiment.py \
  --algo ppo \
  --mdp box2d.cartpole_mdp \
  --normalize_mdp \
  --policy mean_std_nn_policy \
  --vf mujoco_value_function \
  --exp_name ppo_box2d_cartpole \
  --n_parallel 1 \
  --snapshot_mode all \
  --algo_binary_search_penalty False \
  --algo_whole_paths True \
  --algo_batch_size 1000 \
  --algo_max_path_length 100 \
  --algo_n_itr 500 \
  --plot True \
  --algo_plot True \
  --seed 1
  ```

  See [Recipies](https://github.com/dementrock/rllab/wiki/Recipies) for more.

## Visualize a policy

  While running an experiment, all the relevant parameters will be stored in a pickle file under data/EXP_NAME. This file can be used to visualize the policy using the following command:

  ```
  python scripts/sim_policy.py PKL_FILE
  ```

## Contributing

  You should make sure that the contributed code passes pylint static checking
  (or annotate with `pylint disable` flags when necessary).

  Here's my .pylintrc file:

  ```
[MESSAGES CONTROL]

disable=R0913,R0902,C0103,C0111,W0613,R0201,W0603,R0915,R0914,W0141,eval-used

[TYPECHECK]

ignored-modules = numpy, numpy.random, pygame
  ```
