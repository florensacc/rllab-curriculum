import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
from sampler import parallel_sampler
parallel_sampler.init_pool(4)

from policy import AtariRAMPolicy
from algo import PPO
from vf import NoValueFunction
from mdp import AtariMDP
import numpy as np

np.random.seed(0)

if __name__ == '__main__':
    mdp = AtariMDP(rom_path="vendor/atari_roms/pong.bin", obs_type='ram')
    policy = AtariRAMPolicy(mdp, hidden_sizes=[64])
    vf = NoValueFunction()
    algo = PPO(exp_name='atari_pong_10k', max_samples_per_itr=10000, discount=0.99, n_parallel=4, stepsize=0.0016)
    algo.train(mdp, policy, vf)
