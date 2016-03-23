import os

os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,optimizer=None'

from rllab.algo.vpg import VPG
from rllab.algo.tnpg import TNPG
from rllab.algo.ppo import PPO
from rllab.algo.trpo import TRPO
from rllab.env.grid_world_env import GridWorldEnv
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policy.categorical_gru_policy import CategoricalGRUPolicy
from rllab.baseline.zero_baseline import ZeroBaseline
from rllab.misc import ext

algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)


def _test_algo(algo, recurrent=False):
    env = GridWorldEnv()
    if recurrent:
        policy = CategoricalGRUPolicy(env_spec=env.spec, hidden_sizes=(6,))
    else:
        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(6,))
    baseline = ZeroBaseline(env_spec=env.spec)
    algo.train(env=env, policy=policy, baseline=baseline)


def test_vpg():
    _test_algo(VPG(**algo_args))
    _test_algo(VPG(**algo_args), recurrent=True)


def test_ppo():
    args = ext.merge_dict(
        algo_args,
        dict(
            max_penalty_itr=1,
            max_opt_itr=1
        )
    )
    _test_algo(PPO(**args))
    _test_algo(PPO(**args), recurrent=True)


def test_trpo():
    _test_algo(TRPO(n_itr=1))
    _test_algo(TRPO(n_itr=1), recurrent=True)


def test_npg():
    _test_algo(TNPG(n_itr=1))
    _test_algo(TNPG(n_itr=1), recurrent=True)
