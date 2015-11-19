import os
os.environ['TENSORFUSE_MODE'] = 'theano'
from rllab.sampler import parallel_sampler
# Technically, we need to add these initializations below to make sure that the
# processes are created before theano is initialized, so that these processes
# can use the cpu mode while the main process is using the gpu. This can
# probably be avoided when using cgt
parallel_sampler.init_pool(4)
#import plotter
#plotter.init_worker()

from rllab.policy.mujoco_policy import MujocoPolicy
from rllab.algo.ppo import PPO
from rllab.mdp.mujoco.hopper_mdp import HopperMDP
import numpy as np
import tensorfuse.tensor as T

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
    algo = PPO(
        exp_name='hopper_100k',
        samples_per_itr=100000,
        discount=0.99,
        stepsize=0.01,
        plot=False
    )
    algo.train(mdp, policy, vf)
