"""
:author: Vitchyr Pong
"""
import abc
import time

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from sandbox.haoran.mddpg.misc.simple_replay_pool import SimpleReplayPool
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler



class OnlineAlgorithm(RLAlgorithm):
    """
    Online learning algorithm.
    """

    def __init__(
            self,
            env,
            policy,
            exploration_strategy,
            batch_size=64,
            n_epochs=1000,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            soft_target_tau=1e-2,
            max_path_length=1000,
            eval_samples=10000,
            scale_reward=1.,
            render=False,
            eval_epoch_gap=10,
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: A Policy
        :param replay_pool_size: Size of the replay pool
        :param batch_size: Minibatch size for training
        :param n_epochs: Number of epoch
        :param epoch_length: Number of time steps per epoch
        :param min_pool_size: Minimum size of the pool to start training.
        :param discount: Discount factor for the MDP
        :param soft_target_tau: Moving average rate. 1 = update immediately
        :param max_path_length: Maximum path length
        :param eval_samples: Number of time steps to take for evaluation.
        :param scale_reward: How much to multiply the rewards by.
        :param render: Boolean. If True, render the environment.
        :return:
        """
        assert min_pool_size >= 2
        self.env = env
        self.policy = policy
        self.exploration_strategy = exploration_strategy
        self.replay_pool_size = replay_pool_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.tau = soft_target_tau
        self.max_path_length = max_path_length
        self.n_eval_samples = eval_samples
        self.scale_reward = scale_reward
        self.render = render

        self.observation_dim = self.env.observation_space.flat_dim
        self.action_dim = self.env.action_space.flat_dim
        self.rewards_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 1],
                                                  name='rewards')
        self.terminals_placeholder = tf.placeholder(tf.float32,
                                                    shape=[None, 1],
                                                    name='terminals')
        self.pool = SimpleReplayPool(self.replay_pool_size,
                                     self.observation_dim,
                                     self.action_dim)
        self.last_statistics = OrderedDict()
        self.sess = tf.get_default_session() or tf.Session()
        with self.sess.as_default():
            self._init_tensorflow_ops()
        self.es_path_returns = []

        self.eval_sampler = BatchSampler(self)
        self.scope = None
        self.whole_paths = True
        self.eval_epoch_gap = eval_epoch_gap

    def _start_worker(self):
        self.eval_sampler.start_worker()

    @overrides
    def train(self):
        with self.sess.as_default():
            self._init_training()
            self._start_worker()

            observation = self.env.reset()
            self.exploration_strategy.reset()
            itr = 0
            path_length = 0
            path_return = 0
            total_start_time = time.time()
            #WARN: one eval per epoch; one train per itr
            for epoch in range(self.n_epochs):
                logger.push_prefix('Epoch #%d | ' % epoch)
                logger.log("Training started")
                train_start_time = time.time()
                for _ in range(self.epoch_length):
                    action = self.exploration_strategy.get_action(itr,
                                                                  observation,
                                                                  self.policy)
                    if self.render:
                        self.env.render()
                    next_ob, raw_reward, terminal, _ = self.env.step(action)
                    reward = raw_reward * self.scale_reward
                    path_length += 1
                    path_return += reward

                    self.pool.add_sample(observation,
                                         action,
                                         reward,
                                         terminal,
                                         False)
                    if terminal or path_length >= self.max_path_length:
                        self.pool.add_sample(next_ob,
                                             np.zeros_like(action),
                                             np.zeros_like(reward),
                                             np.zeros_like(terminal),
                                             True)
                        observation = self.env.reset()
                        self.exploration_strategy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                    else:
                        observation = next_ob

                    if self.pool.size >= self.min_pool_size:
                        self._do_training()
                    itr += 1

                train_time = time.time() - train_start_time
                logger.log("Training finished. Time: {0}".format(train_time))

                # testing ---------------------------------
                eval_start_time = time.time()
                if self.n_eval_samples > 0:
                    self.evaluate(epoch, self.es_path_returns)
                    self.es_path_returns = []
                eval_time = time.time() - eval_start_time
                logger.log(
                    "Eval time: {0}".format(eval_time))

                # logging --------------------------------
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                logger.record_tabular("time: train",train_time)
                logger.record_tabular("time: eval",eval_time)
                total_time = time.time() - total_start_time
                logger.record_tabular("time: total",total_time)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
            self.env.terminate()
            return self.last_statistics

    def _do_training(self):
        minibatch = self.pool.random_batch(self.batch_size)
        sampled_obs = minibatch['observations']
        sampled_terminals = minibatch['terminals']
        sampled_actions = minibatch['actions']
        sampled_rewards = minibatch['rewards']
        sampled_next_obs = minibatch['next_observations']

        feed_dict = self._update_feed_dict(sampled_rewards,
                                           sampled_terminals,
                                           sampled_obs,
                                           sampled_actions,
                                           sampled_next_obs)
        self.sess.run(self._get_training_ops(), feed_dict=feed_dict)

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
        )

    @abc.abstractmethod
    def _init_tensorflow_ops(self):
        """
        Method to be called in the initialization of the class. After this
        method is called, the train() method should work.
        :return: None
        """
        return

    @abc.abstractmethod
    def _init_training(self):
        """
        Method to be called at the start of training.
        :return: None
        """
        return

    @abc.abstractmethod
    def _get_training_ops(self):
        """
        :return: List of ops to perform when training
        """
        return

    @abc.abstractmethod
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        """
        :return: feed_dict needed for the ops returned by get_training_ops.
        """
        return

    @abc.abstractmethod
    def evaluate(self, epoch, es_path_returns):
        """
        Perform evaluation for this algorithm.

        It's recommended
        :param epoch: The epoch number.
        :param es_path_returns: List of path returns from explorations strategy
        :return: Dictionary of statistics.
        """
        return
