from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from sandbox.rocky.gps.imported.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from sandbox.rocky.gps.imported.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from sandbox.rocky.gps.imported.policy.lin_gauss_init import init_pd
from sandbox.rocky.gps.imported.sample_list import SampleList
from sandbox.rocky.gps.imported.sample import Sample
import numpy as np

env = CartpoleEnv()


class EnvAgent(object):
    def __init__(self, env, T):
        self.env = env
        self.T = T
        self.x0 = np.array([0, 0, 0, 0])
        self.dU = self.env.action_dim
        self.dX = self.env.observation_space.flat_dim
        self.dO = 0

    def sample(self, policy, cond=0, verbose=False):
        obs = self.env.reset()
        assert cond == 0
        import ipdb; ipdb.set_trace()


class EnvCost(object):
    def __init__(self, env):
        self.env = env

agent = EnvAgent(env, T=100)

num_samples = 5

algo = AlgorithmTrajOpt(dict(
    conditions=1,
    agent=agent,
    traj_opt=dict(
        type=TrajOptLQRPython,
    ),
    cost=dict(
        type=EnvCost,
        env=env,
    ),
    init_traj_distr=dict(
        type=init_pd,
        init_var=5.0,
        pos_gains=0.0,
        dQ=env.action_dim,
        T=100,
    )
))

for itr in xrange(100):
    policy = algo.cur[0].traj_distr
    sample_list = []
    for _ in xrange(num_samples):
        sample_list.append(agent.sample(policy, cond=0, verbose=True))
    algo.iteration(sample_lists=[SampleList(sample_list)])
