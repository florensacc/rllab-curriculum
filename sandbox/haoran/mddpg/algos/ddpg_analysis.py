"""
:author: Vitchyr Pong
"""
import time
from collections import OrderedDict

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sandbox.haoran.mddpg.algos.online_algorithm import OnlineAlgorithm
from sandbox.haoran.mddpg.misc.data_processing import create_stats_ordered_dict
from sandbox.haoran.mddpg.misc.rllab_util import split_paths
from sandbox.haoran.myscripts.myutilities import get_true_env
from sandbox.haoran.mddpg.algos.ddpg import DDPG
from sandbox.haoran.mddpg.misc.simple_replay_pool import SimpleReplayPool

from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable

TARGET_PREFIX = "target_"


class DDPGAnalysis(DDPG):
    """
    A more complicated version of DDPG that has various params for debugging /
        analysis.
    Notice that pi uses a smaller batch and we run two sessions to train pi and Q.
    So it can be slower.
    """

    def __init__(
            self,
            qf_batch_size=64,
            policy_batch_size=64,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.qf_batch_size = qf_batch_size
        self.policy_batch_size = policy_batch_size

        super().__init__(**kwargs)

    @overrides
    def _do_training(self):
        self.train_critic = (np.mod(
            self.critic_train_counter,
            self.critic_train_frequency,
        ) == 0)
        self.train_actor = (np.mod(
            self.actor_train_counter,
            self.actor_train_frequency,
        ) == 0)
        self.update_target = (np.mod(
            self.update_target_counter,
            self.update_target_frequency,
        ) == 0)

        if self.train_critic:
            minibatch = self.pool.random_batch(self.qf_batch_size)
            sampled_obs = minibatch['observations']
            sampled_terminals = minibatch['terminals']
            sampled_actions = minibatch['actions']
            sampled_rewards = minibatch['rewards'][:,0] # assume single reward
            sampled_next_obs = minibatch['next_observations']
            feed_dict = self._critic_feed_dict(
                rewards=sampled_rewards,
                terminals=sampled_terminals,
                obs=sampled_obs,
                actions=sampled_actions,
                next_obs=sampled_next_obs,
            )
            ops = [self.train_critic_op]
            if self.debug_mode:
                print("Critic minibatch size:", sampled_obs.shape[0])
                ops.append(tf.Print(
                    self.critic_total_loss,
                    [self.critic_total_loss],
                    message="Critic minibatch loss: ",
                ))
            if self.update_target:
                ops.append(self.update_target_critic_op)
                if self.debug_mode:
                    ops.append(
                        tf.Print(
                            self.tau,
                            [self.tau],
                            message="Update target critic with tau: "
                        )
                    )
            self.sess.run(
                ops,
                feed_dict=feed_dict,
            )

        if self.train_actor:
            minibatch = self.pool.random_batch(self.policy_batch_size)
            sampled_obs = minibatch['observations']
            ops = [self.train_actor_op]
            if self.debug_mode:
                print("Actor minibatch size:", sampled_obs.shape[0])
                ops.append(
                    tf.Print(
                        self.actor_surrogate_loss,
                        [self.actor_surrogate_loss],
                        message="Actor minibatch loss: ",
                    )
                )

            if self.update_target:
                ops.append(self.update_target_actor_op)
                if self.debug_mode:
                    ops.append(tf.Print(
                        self.tau,
                        [self.tau],
                        message="Update target actor with tau: "
                    ))
            self.sess.run(
                ops,
                feed_dict=self._actor_feed_dict(sampled_obs)
            )

        self.critic_train_counter = np.mod(
            self.critic_train_counter + 1,
            self.critic_train_frequency
        )
        self.actor_train_counter = np.mod(
            self.actor_train_counter + 1,
            self.actor_train_frequency,
        )
        self.update_target_counter = np.mod(
            self.update_target_counter + 1,
            self.update_target_frequency,
        )
