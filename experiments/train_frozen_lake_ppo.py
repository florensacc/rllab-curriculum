#!/usr/bin/python
import os
os.environ['TENSORFUSE_MODE'] = 'cgt'
import multiprocessing
from sampler import parallel_sampler
parallel_sampler.init_pool(multiprocessing.cpu_count())
#parallel_sampler.init_pool(1)

from policy.tabular_policy import TabularPolicy
from vf.no_value_function import NoValueFunction
from algo.ppo import PPO
from mdp.frozen_lake_mdp import FrozenLakeMDP
import numpy as np

np.random.seed(0)

if __name__ == '__main__':
    map8x8 = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
    mdp = FrozenLakeMDP(map8x8)
    policy = TabularPolicy(mdp)
    vf = NoValueFunction()
    algo = PPO(
        samples_per_itr=20000,
        max_path_length=100,
        stepsize=0.05,
        adapt_penalty=True,
        discount=0.99,
        initial_penalty=1,
        max_penalty_itr=10,
    )
    algo.train(mdp, policy, vf)
