import sys

import numpy as np
import theano

from rllab.algos.ddpg import DDPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from rlqr import RecurrentLQRPolicy
from common.envs.flexible_cartpole import FlexibleCartpoleEnv

# cartpole = CartpoleEnv()
cartpole = FlexibleCartpoleEnv()
env = normalize(cartpole)

if len(sys.argv) > 1:
    recurrences = int(sys.argv[1])
else:
    recurrences = 10

# obs_dim = cartpole.observation_space.flat_dim
state_dim = 4
Q = np.zeros((state_dim, state_dim)); Q[2,2] = 1
R = np.array([[1]])
policy = RecurrentLQRPolicy(env.spec, Q, R,
        state_dim=4,
        net_hidden_sizes=[100, 100, 100],
        recurrences=recurrences
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
es = OUStrategy(env_spec=env.spec)
qf = ContinuousMLPQFunction(env_spec=env.spec)

def callback():
    r_w, r_h = np.random.rand(2)
    w, h = r_w * 0.1 + 0.05, r_h + 0.5
    cartpole.set_pole_size(w, h)

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
algo.train(callback)
