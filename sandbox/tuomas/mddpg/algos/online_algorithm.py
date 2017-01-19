"""
:author: Vitchyr Pong
"""
import abc
import time
import gtimer as gt

import numpy as np
import tensorflow as tf
from collections import OrderedDict

from sandbox.haoran.mddpg.misc.simple_replay_pool import SimpleReplayPool
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.haoran.myscripts import tf_utils


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
            start_epoch=0,
            n_epochs=1000,
            epoch_length=1000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            soft_target_tau=1e-2,
            max_path_length=1000,
            eval_samples=10000,
            scale_reward=1.,
            scale_reward_annealer=None,
            render=False,
            epoch_full_paths=False,
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
        self.start_epoch = start_epoch
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.min_pool_size = min_pool_size
        self.discount = discount
        self.tau = soft_target_tau
        self.max_path_length = max_path_length
        self.n_eval_samples = eval_samples
        self.scale_reward = scale_reward
        self.scale_reward_annealer = scale_reward_annealer
        self.render = render
        self.epoch_full_paths = epoch_full_paths

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
        self.sess = tf.get_default_session() or tf_utils.create_session()
        with self.sess.as_default():
            self._init_tensorflow_ops()
        self.es_path_returns = []
        self.es_path_lengths = []

        #HT: in general, VectorizedSampler can significantly reduce
        # PolicyExecTime, but not EnvExecTime. The latter consumes more
        # computation in Mujoco tasks, so we prefer BatchSampler
        self.eval_sampler = BatchSampler(self)
        # self.eval_sampler = VectorizedSampler(self,n_envs=16)
        self.scope = None
        self.whole_paths = True

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
            #WARN: one eval per epoch; one train per itr
            gt.rename_root('online algo')
            gt.reset()
            gt.set_def_unique(False)
            total_steps = 0
            for epoch in gt.timed_for(
                range(self.start_epoch, self.start_epoch + self.n_epochs),
                save_itrs=True,
            ):
                self.update_training_settings(epoch)
                logger.push_prefix('Epoch #%d | ' % epoch)
                if self.epoch_full_paths:
                    def is_epoch_finished(t, should_reset):
                        return t >= self.epoch_length and should_reset
                else:
                    def is_epoch_finished(t, should_reset):
                        return t >= self.epoch_length

                t, should_reset = 0, False
                while not is_epoch_finished(t, should_reset):
                    t = t + 1
                    # sampling
                    action = self.exploration_strategy.get_action(itr,
                                                                  observation,
                                                                  self.policy)
                    action.squeeze()
                    if self.render:
                        self.env.render()
                    next_ob, raw_reward, terminal, info = self.env.step(action)
                    reward = raw_reward * self.scale_reward
                    path_length += 1
                    path_return += reward
                    gt.stamp('train: sampling')

                    self.plot_path(rewards=raw_reward,
                                   obs=observation,
                                   actions=action,
                                   info=info)

                    # add experience to replay pool
                    self.pool.add_sample(observation,
                                         action,
                                         reward,
                                         terminal,
                                         False)
                    should_reset = (terminal or
                                    path_length >= self.max_path_length)
                    if should_reset:
                        self.pool.add_sample(next_ob,
                                             np.zeros_like(action),
                                             np.zeros_like(reward),
                                             np.zeros_like(terminal),
                                             True)
                        #self.db.flush()
                        self.plot_path(rewards=raw_reward,
                                       obs=observation,
                                       actions=action,
                                       info=info,
                                       flush=True)

                        observation = self.env.reset()
                        self.exploration_strategy.reset()
                        self.es_path_returns.append(path_return)
                        self.es_path_lengths.append(path_length)
                        path_length = 0
                        path_return = 0
                    else:
                        observation = next_ob
                    gt.stamp('train: fill replay pool')

                    # train
                    if self.pool.size >= self.min_pool_size:
                        self._do_training()
                    itr += 1
                    gt.stamp('train: updates')

                # testing ---------------------------------
                train_info = dict(
                    es_path_returns=self.es_path_returns,
                    es_path_lengths=self.es_path_lengths,
                )
                if self.n_eval_samples > 0:
                    self.evaluate(epoch, train_info)
                    self.es_path_returns = []
                gt.stamp("test")

                # logging --------------------------------
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times = gt.get_times()
                times_itrs = gt.get_times().stamps.itrs
                train_time = np.sum([
                    times_itrs[stamp][-1]
                    for stamp in ["train: sampling",
                    "train: fill replay pool","train: updates"]
                ])
                eval_time = times_itrs["test"][-1]
                total_time = gt.get_times().total
                logger.record_tabular("time: train",train_time)
                logger.record_tabular("time: eval",eval_time)
                logger.record_tabular("time: total",total_time)
                logger.record_tabular("scale_reward", self.scale_reward)
                logger.record_tabular("steps: current epoch", t)
                total_steps += t
                logger.record_tabular("steps: all", total_steps)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
                gt.stamp("logging")

                print(gt.report(
                    include_itrs=False,
                    format_options={
                        'itr_name_width': 30
                    },
                ))
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


        # TH: First train, then finalize. This can be suboptimal.
        self.sess.run(self._get_training_ops(), feed_dict=feed_dict)
        self.sess.run(self._get_finalize_ops(), feed_dict=feed_dict)

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
    def evaluate(self, epoch, train_info):
        """
        Perform evaluation for this algorithm.

        It's recommended
        :param epoch: The epoch number.
        :param es_path_returns: List of path returns from explorations strategy
        :return: Dictionary of statistics.
        """
        return

    @abc.abstractmethod
    def plot_path(self, rewards=None, terminals=None, obs=None, actions=None,
                  next_obs=None, flush=False, info=None):
        """
        Plot training data or apply other postprocessing steps. Called after
        drawing a new training sample.
        :param flush: True if the current sample is the last for the current
            trajectory (either reached terminal state or max path length).
        :return:
        """
        return

    def update_training_settings(self, epoch):
        if self.scale_reward_annealer is not None:
            self.scale_reward = self.scale_reward_annealer.get_new_value(epoch)
