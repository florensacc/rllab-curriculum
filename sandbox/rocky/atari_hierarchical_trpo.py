import sys
sys.path.append(".")

import os
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.misc.console import run_experiment


params = dict(
    mdp=dict(
        _name="openai_atari_mdp",
        rom_name="pong",
        obs_type="ram",
    ),
    policy=dict(
        _name="hierarchical.discrete_subgoal_categorical_policy",
        hidden_sizes=[32, 32],
    ),
    baseline=dict(
        _name="linear_feature_baseline",
    ),
    exp_name="trpo_cartpole",
    algo=dict(
        _name="hierarchical.trpo",
        batch_size=10000,
        whole_paths=True,
        max_path_length=4500,
        n_itr=200,
        discount=0.99,
        step_size=0.01,
    ),
    n_parallel=4,
    snapshot_mode="last",
    seed=1,
)

run_experiment(params)
