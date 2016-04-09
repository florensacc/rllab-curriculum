from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.rl_gym_env import RLGymEnv
from rllab.algos.trpo import TRPO
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = RLGymEnv("SuttonPole-v0")
policy = CategoricalMLPPolicy(env_spec=env.spec)
baseline = GaussianMLPBaseline(env_spec=env.spec)

algo = TRPO(env=env, policy=policy, baseline=baseline, max_path_length=env.horizon, n_itr=10)

run_experiment_lite(
    algo.train(),
    mode='local',
    exp_prefix="rl_gym",
    snapshot_mode="last",
)


