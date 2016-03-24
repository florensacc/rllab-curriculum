from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.algos.vpg import VPG


stub(globals())

env = GridWorldEnv()
policy = CategoricalMLPPolicy(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)
algo = VPG(update_method='adam')


run_experiment_lite(
    algo.train(env=env, policy=policy, baseline=baseline),
    "gridworld_vpg"
)