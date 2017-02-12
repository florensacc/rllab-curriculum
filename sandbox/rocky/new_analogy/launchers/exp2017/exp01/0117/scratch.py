#
# from gpr.envs.stack import Experiment
# expr = Experiment(nboxes=2, horizon=500)
# env = expr.make(task_id=[[1, "top", 0]])
# import ipdb; ipdb.set_trace()
# print(env.world.xml)
#
import multiprocessing

import itertools

from gpr.envs.fetch_rl import Experiment
# from sandbox.rocky.new_analogy.envs.stack_fetch import Experiment
from sandbox.rocky.new_analogy.gpr_ext.pi2_fast import pi2
from sandbox.rocky.s3.resource_manager import resource_manager

# num_substeps = 1#5
horizon = 100#300 // num_substeps

expr = Experiment(nboxes=2, horizon=horizon)
env = expr.make(task_id=[[1, 0]])
import gpr.trajectory
from gpr import Trajectory

gpr.trajectory.optimizers["pi2_fast"] = pi2

from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics
ffd = FastForwardDynamics(env, n_parallel=multiprocessing.cpu_count())

optimizer_params = expr.optimizer_params._replace(
    optimizer="pi2_fast",
    save_intermediate=True,
    mpc_horizon=40,
    mpc_steps=20,
    particles=100,
    # mpc_horizon=40,
    # mpc_steps=20,
    # particles=300,
    # skip=4,
    # max_kl=100,
    extras=dict(ffd=ffd, discount=1.),#0.9),
)

import gym.spaces.prng

# for seed in itertools.count(start=, step=n_workers):
#     logger.log("Computing traj {0}".format(seed))
#     random.seed(seed)
#     np.random.seed(seed)
env.seed(0)
gym.spaces.prng.seed(0)
trajectory = Trajectory(env)
trajectory.optimize(optimizer_params)


# resource_manager.register_file("stack_bc_v0", "/tmp/params.pkl")
