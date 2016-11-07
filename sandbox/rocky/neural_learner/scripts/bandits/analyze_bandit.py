import itertools
import joblib
import pydotplus
import tensorflow as tf
import numpy as np
from sklearn.tree import export_graphviz

from rllab.envs.base import Env
from rllab.sampler.utils import rollout
from sandbox.rocky.neural_learner.envs.multi_env import BIG
from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.spaces import Product, Discrete, Box

with tf.Session() as tf:
    # (5, 10)
    # file = "data/s3/mab-17/mab-17_2016_10_30_17_08_35_0031/params.pkl"

    # (5, 500)
    file = "data/s3/mab-17/mab-17_2016_10_30_17_08_35_0041/params.pkl"
    data = joblib.load(file)
    # env = data['env']
    policy = data['policy']


    class FixedEnv(Env):
        def __init__(self, reward_sequence, n_arms):
            self.reward_sequence = reward_sequence
            self.n_arms = n_arms
            self.t = 0

        def reset(self):
            self.t = 0
            return (0, 0, 0, 1)

        @property
        def observation_space(self):
            return Product(
                Discrete(1),
                Discrete(self.n_arms),
                Box(low=-BIG, high=BIG, shape=(1,)),
                Box(low=0, high=1, shape=(1,))
            )

        @property
        def action_space(self):
            return Discrete(self.n_arms)

        def step(self, action):
            rew = self.reward_sequence[self.t]
            self.t += 1
            return (0, action, rew, 0), rew, False, dict()


    class VectorizedFixedEnv(object):
        def __init__(self, horizon, n_arms):
            self.horizon = horizon
            self.n_arms = n_arms
            self.reward_sequences = np.asarray(list(itertools.product([0, 1], repeat=horizon)))
            self.n_envs = len(self.reward_sequences)

        def reset(self, *args, **kwargs):
            self.t = 0
            return [(0, 0, 0, 1)] * self.n_envs

        def step(self, action_n, max_path_length):
            rew = self.reward_sequences[:, min(self.t, self.horizon - 1)]

            obs = [0] * self.n_envs
            dones = [self.t == self.horizon - 1] * self.n_envs


            self.t += 1

            return list(zip(obs, action_n, rew, dones)), rew, dones, dict()

    horizon = 5

    env = FixedEnv(reward_sequence=[0] * 10, n_arms=5)
    vec_env = VectorizedFixedEnv(horizon=horizon, n_arms=5)

    sampler = VectorizedSampler(env=env, policy=policy, n_envs=2 ** horizon, vec_env=vec_env)

    paths = sampler.obtain_samples(itr=0, max_path_length=horizon, batch_size=horizon * (2 ** horizon))

    joblib.dump(paths, "paths.pkl", compress=True)

    # class Tree(object):
    #
    #     def __init__(self, paths):
    #         self.paths = paths
    #         self.tree_ = self
    #         self.criterion = None
    #
    #
    # dot_data = export_graphviz(Tree(paths))
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # # pydotplus.Graph
    # graph.write_pdf("aa.dpf")

    # import ipdb;
    #
    # ipdb.set_trace()

    # while True:
    # path = rollout(env=FixedEnv(reward_sequence=[0] * 10, n_arms=5), agent=policy,
    #                max_path_length=env.wrapped_env.n_episodes)
    # print("Actions:", np.argmax(path["actions"], axis=1) + 1)
    # # arm_means = env.wrapped_env.wrapped_env.executor.arm_means[0]
    # print("Rewards:", path["rewards"])  # np.argmax(path["actions"], axis=1))
    # # print("Arm means:", arm_means)
    # # print("Best arm:", np.argmax(arm_means) + 1)
    # # import ipdb;
    # #
    # # ipdb.set_trace()
