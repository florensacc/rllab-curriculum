from rllab.algos.nop import NOP
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.uniform_control_policy import UniformControlPolicy
from sandbox.young_clgan.envs.action_limited_env import ActionLimitedEnv
from sandbox.young_clgan.envs.arm3d.arm3d_disc_robust_env import Arm3dDiscRobustEnv

# env = normalize(CartpoleEnv())
env = Arm3dDiscRobustEnv()
# env = ActionLimitedEnv(Arm3dDiscRobustEnv())

policy = UniformControlPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
)

baseline = ZeroBaseline(env_spec=env.spec)

algo = NOP(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=200,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    plot=True
)
algo.train()
