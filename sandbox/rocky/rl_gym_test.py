from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.rl_gym_env import RLGymEnv
from rllab.algos.ppo import PPO
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = RLGymEnv("SuttonPole-v0")
policy = CategoricalMLPPolicy(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)
algo = PPO(env=env, policy=policy, baseline=baseline, max_path_length=env.horizon, n_itr=20)

run_experiment_lite(
    algo.train(),
    exp_prefix="rl_gym",
    snapshot_mode="last",
)


