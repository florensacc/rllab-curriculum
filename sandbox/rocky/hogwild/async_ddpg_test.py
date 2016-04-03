from sandbox.rocky.hogwild.async_ddpg import AsyncDDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = CartpoleEnv()
policy = DeterministicMLPPolicy(env_spec=env.spec)
qf = ContinuousMLPQFunction(env_spec=env.spec)
es = OUStrategy(env_spec=env.spec)
algo = AsyncDDPG(env=env, policy=policy, qf=qf, es=es)

run_experiment_lite(
    algo.train(),
    exp_prefix="async_ddpg"
)
