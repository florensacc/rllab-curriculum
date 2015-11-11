import os
os.environ['TENSORFUSE_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu'
from sampler import parallel_sampler
# Technically, we need to add these initializations below to make sure that the
# processes are created before theano is initialized, so that these processes
# can use the cpu mode while the main process is using the gpu. This can
# probably be avoided when using cgt
parallel_sampler.init_pool(4)
#import plotter
#plotter.init_worker()

from policy.mujoco_policy import MujocoPolicy
from algo.ppo import PPO
from mdp.swimmer_mdp import SwimmerMDP
import numpy as np
import tensorfuse.tensor as T
from vf.no_value_function import NoValueFunction

#np.random.seed(0)

class SwimmerValueFunction(object):

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
    mdp = SwimmerMDP()
    policy = MujocoPolicy(mdp, hidden_sizes=[30, 30])
    vf = SwimmerValueFunction()
    algo = PPO(
        exp_name='swimmer_50k_new',
        samples_per_itr=50000,
        max_path_length=500,
        discount=0.99,
        stepsize=0.01,
        plot=False#True#False#True
    )
    algo.train(mdp, policy, vf)
