import numpy as np
import theano

from rllab.algos.ddpg import DDPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from rlqr import RecurrentLQRPolicy

env = normalize(CartpoleEnv())
Q = np.zeros((4,4)); Q[2,2] = 1
R = np.array([[1]])
policy = RecurrentLQRPolicy(env.spec, Q, R)
baseline = LinearFeatureBaseline(env_spec=env.spec)
es = OUStrategy(env_spec=env.spec)
qf = ContinuousMLPQFunction(env_spec=env.spec)

algo = DDPG(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    batch_size=32,
    max_path_length=100,
    epoch_length=1000,
    min_pool_size=10000,
    n_epochs=1000,
    discount=0.99,
    scale_reward=0.01,
    qf_learning_rate=1e-3,
    policy_learning_rate=1e-4
)
algo.train()
