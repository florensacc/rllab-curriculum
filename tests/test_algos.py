import os

os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,optimizer=None'

from rllab.algo.vpg import VPG
from rllab.algo.tnpg import TNPG
from rllab.algo.ppo import PPO
from rllab.algo.trpo import TRPO
from rllab.env.grid_world_env import GridWorldEnv
from rllab.env.box2d.cartpole_env import CartpoleEnv
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policy.categorical_gru_policy import CategoricalGRUPolicy
from rllab.policy.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baseline.zero_baseline import ZeroBaseline
from rllab.misc import ext

TEST_RECURRENT = False  # True

algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)


def _test_algo(algo):
    env_policy = [
        (GridWorldEnv, CategoricalMLPPolicy),
        (CartpoleEnv, GaussianMLPPolicy)
        # (GridWorldEnv, CategoricalGRUPolicy),
    ]
    if TEST_RECURRENT:
        env_policy.append(
            (GridWorldEnv, CategoricalGRUPolicy),
        )
    for env_cls, policy_cls in env_policy:
        print "Testing %s with %s" % (str(env_cls), str(policy_cls))
        env = env_cls()
        policy = policy_cls(env_spec=env.spec, hidden_sizes=(6,))
        baseline = ZeroBaseline(env_spec=env.spec)
        algo.train(env=env, policy=policy, baseline=baseline)


def test_vpg():
    args = ext.merge_dict(
        algo_args,
        dict(
            optimizer_args=dict(
                max_penalty_itr=1,
                max_opt_itr=1
            )
        )
    )
    _test_algo(VPG(**args))


def test_ppo():
    args = ext.merge_dict(
        algo_args,
        dict(
            optimizer_args=dict(
                max_penalty_itr=1,
                max_opt_itr=1
            )
        )
    )
    _test_algo(PPO(**args))


def test_trpo():
    args = ext.merge_dict(
        algo_args,
        dict(
            optimizer_args=dict(
                cg_iters=1,
            )
        )
    )
    _test_algo(TRPO(**args))


def test_tnpg():
    args = ext.merge_dict(
        algo_args,
        dict(
            optimizer_args=dict(
                cg_iters=1,
            )
        )
    )
    _test_algo(TNPG(**args))
