


from sandbox.rocky.tf.algos.a3c import A3C
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.regressors.deterministic_mlp_regressor import DeterministicMLPRegressor
from rllab.misc.instrument import stub, run_experiment_lite
import tensorflow as tf

stub(globals())

# env = TfEnv(normalize(CartpoleEnv()))
# env = TfEnv(normalize(CartpoleSwingupEnv()))
# env = TfEnv(normalize(HalfCheetahEnv()))
env = TfEnv(normalize(HopperEnv()))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32),
    std_parametrization='exp'
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = A3C(
    env=env,
    policy=policy,
    scale_reward=0.01,
    n_workers=1,
    critic_network=DeterministicMLPRegressor(
        name="critic_regressor",
        output_dim=1,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.identity,
        hidden_sizes=(32, 32),
        input_shape=(env.observation_space.flat_dim,),
    ),
    max_path_length=500,
    epoch_length=1000,
)
run_experiment_lite(
    algo.train(),
    seed=1,
)