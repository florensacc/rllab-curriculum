import multiprocessing

from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.new_analogy.policies.residual_policy import ResidualPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import pickle

"""
Behavior cloning on fetch using noisy trajectories
"""

MODE = "local_docker"
N_PARALLEL = 1


def run_task(vv):
    from gpr_package.bin import tower_fetch_policy as tower
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.algos.ff_bc_trainer import Trainer
    from sandbox.rocky.s3.resource_manager import resource_manager
    import os
    import subprocess
    import numpy as np

    task_id = tower.get_task_from_text("ab")
    expr = tower.SimFetch(nboxes=2, horizon=2500, mocap=True, obs_type="full_state")
    env = expr.make(task_id)
    policy = tower.FetchPolicy(task_id=task_id)
    from rllab.envs.gym_env import convert_gym_space
    obs_space = convert_gym_space(env.observation_space)

    file_name = resource_manager.get_file("tower_fetch_ab_on_policy_round_1")
    with open(file_name, "rb") as f:
        paths = pickle.load(f)

    with multiprocessing.Pool() as pool:
        for idx, path in enumerate(paths):
            print(idx)
            obs = obs_space.unflatten_n(path["observations"])
            actions = np.asarray(pool.map(policy.get_action, obs))
            path["taught_actions"] = actions

    resource_manager.register_data("tower_fetch_on_policy_round_1_ab_annotated", pickle.dumps(paths))

kwargs = dict(
    use_cloudpickle=True,
    mode=MODE,
    use_gpu=True,
    snapshot_mode="last",
    sync_all_data_node_to_s3=False,
    n_parallel=N_PARALLEL,
    env=dict(CUDA_VISIBLE_DEVICES="4", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
    variant=dict(),
)

if MODE == "local":
    del kwargs["env"]["PYTHONPATH"]  # =
else:
    kwargs = dict(
        kwargs,
        docker_image="dementrock/rocky-rllab3-gpr-gpu-pascal:20170115",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    )

run_experiment_lite(
    run_task,
    **kwargs
)
