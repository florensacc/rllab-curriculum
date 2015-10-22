import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
from sampler import parallel_sampler
parallel_sampler.init_pool(1)

from policy import MujocoPolicy
from algo import PPO
from mdp import HopperMDP
import numpy as np
import cgtcompat.tensor as T

np.random.seed(0)

class HopperValueFunction(object):

    def __init__(self):
        self.coeffs = None

    def get_param_values(self):
        return self.coeffs

    def set_param_values(self, val):
        self.coeffs = val

    def _features(self, path):
        o = np.clip(path["observations"], -10,10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1,1)/100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((l,1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]

    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self.coeffs)

if __name__ == '__main__':
    mdp = HopperMDP()
    policy = MujocoPolicy(mdp, hidden_sizes=[32, 32])
    vf = HopperValueFunction()
    algo = PPO(exp_name='hopper_10k', max_samples_per_itr=10000, discount=0.98, n_parallel=4, stepsize=0.0016)
    algo.train(mdp, policy, vf)
