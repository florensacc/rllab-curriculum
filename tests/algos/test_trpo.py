from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env, Step
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.algos.trpo import TRPO
from rllab.spaces.box import Box
import lasagne.nonlinearities
import numpy as np
import theano.tensor as TT


class DummyEnv(Env):
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(1,))

    @property
    def action_space(self):
        return Box(low=-5.0, high=5.0, shape=(1,))

    def reset(self):
        return np.zeros(1)

    def step(self, action):
        return Step(observation=np.zeros(1), reward=0, done=True)


def test_trpo_nan():
    env = DummyEnv()
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_nonlinearity=lambda x: TT.max(x, 0),
        hidden_sizes=(1,))
    baseline = ZeroBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env, policy=policy, baseline=baseline, n_itr=1, batch_size=1000, max_path_length=100,
        step_size=0.001
    )
    algo.train()
    assert not np.isnan(np.sum(policy.get_param_values()))
