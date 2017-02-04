import multiprocessing

import cloudpickle

import random
from rllab import config

import itertools

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.s3.resource_manager import resource_manager

import numpy as np

n_workers = 50
horizon = 1000
n_boxes = 2
task_id = [[1, "top", 0]]
mpc_horizon = 40
mpc_steps = 20


def run_task(vv):
    from gpr import Trajectory
    from sandbox.rocky.new_analogy.gpr_ext.pi2_fast import pi2
    worker_id = vv["worker_id"]

    from gpr.envs.stack import Experiment
    expr = Experiment(nboxes=n_boxes, horizon=horizon)
    env = expr.make(task_id=task_id)
    import gpr.trajectory

    gpr.trajectory.optimizers["pi2_fast"] = pi2

    from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics
    ffd = FastForwardDynamics(env, n_parallel=multiprocessing.cpu_count())

    optimizer_params = expr.optimizer_params._replace(
        optimizer="pi2_fast",
        save_intermediate=False,
        mpc_horizon=mpc_horizon,
        mpc_steps=mpc_steps,
        extras=dict(ffd=ffd),
    )

    for seed in itertools.count(start=worker_id, step=n_workers):
        logger.log("Computing traj {0}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        trajectory = Trajectory(env)
        trajectory.optimize(optimizer_params)
        resource_manager.register_data("stack_ab_trajs_v0/traj_{0}.pkl".format(seed), cloudpickle.dumps(trajectory))


for worker_id in range(n_workers):
    config.AWS_INSTANCE_TYPE = "c4.2xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = random.choice(
        ['us-west-1', 'us-east-1', 'us-west-2']
    )
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="stack-pi2-3",
        variant=dict(worker_id=worker_id),
        terminate_machine=True,
        mode="ec2",
        env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
        sync_log_on_termination=False,
        periodic_sync=False,
        sync_all_data_node_to_s3=False,
    )
