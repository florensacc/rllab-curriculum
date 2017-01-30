from sandbox.haoran.mddpg.algos.online_algorithm import OnlineAlgorithm
from sandbox.haoran.mddpg.misc.data_processing import create_stats_ordered_dict
from sandbox.haoran.mddpg.misc.rllab_util import split_paths
from sandbox.haoran.mddpg.policies.mnn_policy import MNNPolicy, MNNStrategy
from sandbox.haoran.mddpg.misc.simple_replay_pool import SimpleReplayPool
from sandbox.rocky.tf.misc.tensor_utils import flatten_tensor_variables

from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.envs.proxy_env import ProxyEnv

import time
from collections import OrderedDict
import numpy as np
import tensorflow as tf

TARGET_PREFIX = "target_"


class OnlyTrainQf(OnlineAlgorithm):
    """
    Only train the q function
    The sampling policy is stochastic
    """

    def __init__(
            self,
            env,
            policy,
            exploration_strategy,
            qf,
            qf_learning_rate=1e-3,
            Q_weight_decay=0.,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: a multiheaded policy
        :param qf: QFunctions that is Serializable
        :param qf_learning_rate: Learning rate of the critic
        :param Q_weight_decay: How much to decay the weights for Q
        :return:
        """
        self.qf = qf
        self.critic_learning_rate = qf_learning_rate
        self.Q_weight_decay = Q_weight_decay

        super().__init__(env, policy, exploration_strategy, **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
        self.target_qf = self.qf.get_copy(
            scope_name=TARGET_PREFIX + self.qf.scope_name,
        )
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        self.target_qf.sess = self.sess
        self._init_critic_ops()
        self._init_target_ops()
        self.sess.run(tf.global_variables_initializer())

    def _init_critic_ops(self):
        #NOTE: we could also sample multiple actions from the policy to form
        # the target

        self.ys = self.rewards_placeholder + \
            (1. - self.terminals_placeholder) * self.discount * \
            self.target_qf.output

        self.critic_loss = tf.reduce_mean(
            tf.square(
                tf.sub(self.ys, self.qf.output)))
        self.Q_weights_norm = tf.reduce_sum(
            tf.pack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(only_regularizable=True)]
            ),
            name='weights_norm'
        )
        self.critic_total_loss = (
            self.critic_loss + self.Q_weight_decay * self.Q_weights_norm)
        self.train_critic_op = tf.train.AdamOptimizer(
            self.critic_learning_rate).minimize(
            self.critic_total_loss,
            var_list=self.qf.get_params_internal())


    def _init_target_ops(self):
        # Since target Q functions share weights, only update one of them
        critic_vars = self.qf.get_params_internal()
        target_critic_vars = self.target_qf.get_params_internal()
        assert len(critic_vars) == len(target_critic_vars)
        self.update_target_critic_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_critic_vars, critic_vars)
        ]

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_qf.set_param_values(self.qf.get_param_values())

    @overrides
    def _get_training_ops(self):
        train_ops = [
            self.train_critic_op,
            self.update_target_critic_op,
        ]
        return train_ops

    def _get_finalize_ops(self):
        return []

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        sampled_next_actions = self.policy.get_actions(next_obs)
        critic_feed = self._critic_feed_dict(rewards,
                                             terminals,
                                             obs,
                                             actions,
                                             next_obs,
                                             next_actions)
        return {**critic_feed}

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs,
        next_actions):
        feed = {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: actions,
            self.target_qf.observations_placeholder: next_obs,
            self.target_qf.actions_placeholder: next_actions,
        }
        return feed

    @overrides
    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = self.eval_sampler.obtain_samples(
            itr=epoch,
            batch_size=self.n_eval_samples,
            max_path_length=self.max_path_length,
        )
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        # Compute statistics
        (
            qf_loss,
            qf_outputs,
            target_qf_outputs,
            ys,
        ) = self.sess.run(
            [
                self.critic_loss,
                self.qf.output,
                self.target_qf_outputs,
                self.ys,
            ],
            feed_dict=feed_dict)
        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount)
             for path in paths]
        )
        returns = np.asarray([sum(path["rewards"]) for path in paths])
        rewards = np.hstack([path["rewards"] for path in paths])

        # Log statistics
        self.last_statistics.update(OrderedDict([
            ('Epoch', epoch),
            ('CriticLoss', qf_loss),
            ('AverageDiscountedReturn', average_discounted_return),
        ]))
        self.last_statistics.update(create_stats_ordered_dict('Ys', ys))
        self.last_statistics.update(create_stats_ordered_dict('QfOutput',
                                                         qf_outputs))
        self.last_statistics.update(create_stats_ordered_dict('TargetQfOutput',
                                                         target_qf_outputs))
        self.last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        self.last_statistics.update(create_stats_ordered_dict('returns', returns))

        if len(es_path_returns) == 0 and epoch == 0:
            es_path_returns = [0]
        if len(es_path_returns) > 0:
            # if eval is too often, training may not even have collected a full
            # path
            train_returns = np.asarray(es_path_returns) / self.scale_reward
            self.last_statistics.update(create_stats_ordered_dict(
                'TrainingReturns', train_returns))

        true_env = self.env
        while isinstance(true_env,ProxyEnv):
            true_env = true_env._wrapped_env
        env_stats = true_env.log_stats(paths)
        self.last_statistics.update(env_stats)

        for key, value in self.last_statistics.items():
            logger.record_tabular(key, value)


    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            env=self.env,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )
