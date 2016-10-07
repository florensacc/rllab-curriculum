from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
from rllab.misc.ext import set_seed
import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.baselines.zero_baseline import ZeroBaseline
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_rnn_policy import CategoricalRNNPolicy
from sandbox.rocky.tf.policies.rnn_utils import NetworkType
from rllab.sampler.utils import rollout
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

env = DoomGoalFindingMazeEnv()


vec_env = env.vec_env_executor(10)
set_seed(0)
print(np.linalg.norm(vec_env.reset()))
set_seed(0)
print(np.linalg.norm(vec_env.reset()))

env = TfEnv(env)
# print(np.linalg.norm(env.reset()))

policy = CategoricalRNNPolicy(
    env_spec=env.spec,
    hidden_nonlinearity=tf.nn.relu,#getattr(tf.nn, v["nonlinearity"]),
    weight_normalization=True,#v["weight_normalization"],
    layer_normalization=False,#v["layer_normalization"],
    network_type=NetworkType.GRU,
    state_include_action=False,
    feature_network=ConvNetwork(
        name="embedding_network",
        input_shape=env.observation_space.shape,
        output_dim=256,
        hidden_sizes=(),
        conv_filters=(16, 32),
        conv_filter_sizes=(5, 5),
        conv_strides=(4, 2),
        conv_pads=('VALID', 'VALID'),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.relu,
        weight_normalization=True,#v["weight_normalization"],
        batch_normalization=False,#v["batch_normalization"],
    ),
    name="policy"
)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(np.linalg.norm(policy.get_param_values()))
    baseline = ZeroBaseline(env.spec)
    algo = PPOSGD(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=1000,#v["batch_size"],
                max_path_length=100,
    )
    set_seed(0)
    print(np.linalg.norm(rollout(env, policy, max_path_length=100)["observations"]))
    set_seed(0)
    print(np.linalg.norm(rollout(env, policy, max_path_length=100)["observations"]))
    sampler = VectorizedSampler(algo, n_envs=10)
    sampler.start_worker()
    set_seed(0)
    print(np.linalg.norm(np.concatenate([p["observations"] for p in sampler.obtain_samples(0, max_path_length=100,
                                                                                 batch_size=1000)])))
    set_seed(0)
    print(np.linalg.norm(np.concatenate([p["observations"] for p in sampler.obtain_samples(0, max_path_length=100,
                                                                                 batch_size=1000)])))
