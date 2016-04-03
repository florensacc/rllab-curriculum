from sandbox.rocky.hogwild.async_ddpg import AsyncDDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = SwimmerEnv()#SwimmerEnv()#HalfCheetahEnv()#CartpoleEnv()
policy = DeterministicMLPPolicy(env_spec=env.spec)#, hidden_sizes=(400, 300))
qf = ContinuousMLPQFunction(env_spec=env.spec)#, hidden_sizes=(400, 300))
es = OUStrategy(env_spec=env.spec)
algo = AsyncDDPG(env=env, policy=policy, qf=qf, es=es, n_workers=4, scale_reward=0.1, qf_learning_rate=1e-3,
                 max_path_length=500, policy_learning_rate=1e-4)

run_experiment_lite(
    algo.train(),
    exp_prefix="async_ddpg"
)
