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
from rllab.policy.gaussian_gru_policy import GaussianGRUPolicy
from rllab.baseline.zero_baseline import ZeroBaseline
from rllab.misc import ext
from nose2 import tools


common_algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)

algo_args = {
    VPG: dict(),
    TNPG: dict(
        optimizer_args=dict(
            cg_iters=1,
        ),
    ),
    TRPO: dict(
        optimizer_args=dict(
            cg_iters=1,
        ),
    ),
    PPO: dict(
        optimizer_args=dict(
            max_penalty_itr=1,
            max_opt_itr=1
        ),
    ),
}

cases = []
for algo in [VPG, TNPG, PPO, TRPO]:
    cases.extend([
        (algo, GridWorldEnv, CategoricalMLPPolicy),
        (algo, CartpoleEnv, GaussianMLPPolicy),
        (algo, GridWorldEnv, CategoricalGRUPolicy),
        (algo, CartpoleEnv, GaussianGRUPolicy),
    ])


@tools.params(*cases)
def test_algo(algo_cls, env_cls, policy_cls):
    print "Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__, policy_cls.__name__)
    algo = algo_cls(**ext.merge_dict(common_algo_args, algo_args.get(algo_cls, dict())))
    env = env_cls()
    policy = policy_cls(env_spec=env, hidden_sizes=(6,))
    baseline = ZeroBaseline(env_spec=env.spec)
    algo.train(env=env, policy=policy, baseline=baseline)
