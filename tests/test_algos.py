import os

from rllab.algos.cem import CEM
from rllab.algos.cma_es import CMAES
from rllab.algos.erwr import ERWR

os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'

from rllab.algos.vpg import VPG
from rllab.algos.tnpg import TNPG
from rllab.algos.ppo import PPO
from rllab.algos.trpo import TRPO
from rllab.algos.reps import REPS
from rllab.algos.dpg import DPG
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc import ext
from nose2 import tools

common_batch_algo_args = dict(
    n_itr=1,
    batch_size=1000,
    max_path_length=100,
)

algo_args = {
    VPG: ext.merge_dict(common_batch_algo_args, dict()),
    TNPG: ext.merge_dict(common_batch_algo_args, dict(
        optimizer_args=dict(
            cg_iters=1,
        ),
    )),
    TRPO: ext.merge_dict(common_batch_algo_args, dict(
        optimizer_args=dict(
            cg_iters=1,
        ),
    )),
    PPO: dict(
        optimizer_args=dict(
            max_penalty_itr=1,
            max_opt_itr=1
        ),
    ),
    REPS: dict(
        max_opt_itr=1,
    ),
    DPG: dict(
        n_epochs=1,
        epoch_length=100,
        batch_size=32,
        min_pool_size=50,
        replay_pool_size=1000,
        eval_samples=100,
    ),
}

cases = []
for algo in [VPG, TNPG, PPO, TRPO, CEM, CMAES, ERWR, REPS]:
    cases.extend([
        (algo, GridWorldEnv, CategoricalMLPPolicy),
        (algo, CartpoleEnv, GaussianMLPPolicy),
        (algo, GridWorldEnv, CategoricalGRUPolicy),
        (algo, CartpoleEnv, GaussianGRUPolicy),
    ])
for algo in [DPG]:
    cases.extend([
        (algo, CartpoleEnv, DeterministicMLPPolicy),
    ])


@tools.params(*cases)
def test_algo(algo_cls, env_cls, policy_cls):
    print "Testing %s, %s, %s" % (algo_cls.__name__, env_cls.__name__, policy_cls.__name__)
    algo = algo_cls(**ext.merge_dict(common_algo_args, algo_args.get(algo_cls, dict())))
    env = env_cls()
    policy = policy_cls(env_spec=env.spec, hidden_sizes=(6,))
    baseline = ZeroBaseline(env_spec=env.spec)
    algo.train(env=env, policy=policy, baseline=baseline)
