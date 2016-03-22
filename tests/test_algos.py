import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE'

from rllab.algo.vpg import VPG
from rllab.algo.npg import NPG
from rllab.algo.ppo import PPO
from rllab.algo.trpo import TRPO
from rllab.env.grid_world_env import GridWorldEnv
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policy.categorical_gru_policy import CategoricalGRUPolicy
from rllab.baseline.zero_baseline import ZeroBaseline


def _test_algo(algo, recurrent=False):
    env = GridWorldEnv()
    if recurrent:
        policy = CategoricalGRUPolicy(env_spec=env.spec)
    else:
        policy = CategoricalMLPPolicy(env_spec=env.spec)
    baseline = ZeroBaseline(env_spec=env.spec)
    algo.train(env=env, policy=policy, baseline=baseline)


def test_vpg():
    _test_algo(VPG(n_itr=1))
    # _test_algo(VPG(n_itr=1), recurrent=True)


def test_ppo():
    _test_algo(PPO(n_itr=1))
    # _test_algo(PPO(n_itr=1), recurrent=True)


def test_trpo():
    _test_algo(TRPO(n_itr=1))
    # _test_algo(TRPO(n_itr=1), recurrent=True)


def test_npg():
    _test_algo(NPG(n_itr=1))
    # _test_algo(NPG(n_itr=1), recurrent=True)
