"""
:author: Vitchyr Pong
"""
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from sandbox.haoran.mddpg.algos.online_algorithm import OnlineAlgorithm
from sandbox.haoran.mddpg.misc.data_processing import create_stats_ordered_dict
from sandbox.haoran.mddpg.misc.rllab_util import split_paths

from sandbox.haoran.mddpg.misc.simple_replay_pool import SimpleReplayPool
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides

TARGET_PREFIX = "target_"


class DDPG(OnlineAlgorithm):
    """
    Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy,
            qf,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            Q_weight_decay=0.,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param qf_learning_rate: Learning rate of the critic
        :param policy_learning_rate: Learning rate of the actor
        :param Q_weight_decay: How much to decay the weights for Q
        :return:
        """
        self.qf = qf
        self.critic_learning_rate = qf_learning_rate
        self.actor_learning_rate = policy_learning_rate
        self.Q_weight_decay = Q_weight_decay

        super().__init__(env, policy, exploration_strategy, **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.initialize_all_variables())
        self.target_policy = self.policy.get_copy(
            scope_name=TARGET_PREFIX + self.policy.scope_name,
        )
        self.target_qf = self.qf.get_copy(
            scope_name=TARGET_PREFIX + self.qf.scope_name,
            action_input=self.target_policy.output
        )
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        self.target_qf.sess = self.sess
        self.target_policy.sess = self.sess
        self._init_critic_ops()
        self._init_actor_ops()
        self._init_target_ops()
        self.sess.run(tf.initialize_all_variables())

    def _init_critic_ops(self):
        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * self.target_qf.output)
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

    def _init_actor_ops(self):
        # To compute the surrogate loss function for the critic, it must take
        # as input the output of the actor. See Equation (6) of "Deterministic
        # Policy Gradient Algorithms" ICML 2014.
        self.critic_with_action_input = self.qf.get_weight_tied_copy(
            self.policy.output)
            # remember that the critic takes no action input at the beginning
        self.actor_surrogate_loss = - tf.reduce_mean(
            self.critic_with_action_input.output)
        self.train_actor_op = tf.train.AdamOptimizer(
            self.actor_learning_rate).minimize(
            self.actor_surrogate_loss,
            var_list=self.policy.get_params_internal())

    def _init_target_ops(self):
        actor_vars = self.policy.get_params_internal()
        critic_vars = self.qf.get_params_internal()
        target_actor_vars = self.target_policy.get_params_internal()
        target_critic_vars = self.target_qf.get_params_internal()
        assert len(actor_vars) == len(target_actor_vars)
        assert len(critic_vars) == len(target_critic_vars)

        self.update_target_actor_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_actor_vars, actor_vars)]
        self.update_target_critic_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_critic_vars, critic_vars)]

    def _get_finalize_ops(self):
        # returning an emptyr list will induce error in tensorflow,
        # so return a useless operation
        return [tf.constant(1)]

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_qf.set_param_values(self.qf.get_param_values())
        self.target_policy.set_param_values(self.policy.get_param_values())

    @overrides
    def _get_training_ops(self):
        return [
            self.train_actor_op,
            self.train_critic_op,
            self.update_target_critic_op,
            self.update_target_actor_op,
        ]

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        critic_feed = self._critic_feed_dict(rewards,
                                             terminals,
                                             obs,
                                             actions,
                                             next_obs)
        actor_feed = self._actor_feed_dict(obs)
        return {**critic_feed, **actor_feed}

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: actions,
            self.target_qf.observations_placeholder: next_obs,
            self.target_policy.observations_placeholder: next_obs,
        }

    def _actor_feed_dict(self, obs):
        return {
            self.critic_with_action_input.observations_placeholder: obs,
            self.policy.observations_placeholder: obs,
        }

    @overrides
    def evaluate(self, epoch, train_info):
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
            policy_loss,
            qf_loss,
            policy_outputs,
            target_policy_outputs,
            qf_outputs,
            target_qf_outputs,
            ys,
        ) = self.sess.run(
            [
                self.actor_surrogate_loss,
                self.critic_loss,
                self.policy.output,
                self.target_policy.output,
                self.qf.output,
                self.target_qf.output,
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
            ('PolicySurrogateLoss', policy_loss),
            #HT: why are the policy outputs info helpful?
            ('PolicyMeanOutput', np.mean(policy_outputs)),
            ('PolicyStdOutput', np.std(policy_outputs)),
            ('TargetPolicyMeanOutput', np.mean(target_policy_outputs)),
            ('TargetPolicyStdOutput', np.std(target_policy_outputs)),
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

        es_path_returns = train_info["es_path_returns"]
        if len(es_path_returns) == 0 and epoch == 0:
            es_path_returns = [0]
        if len(es_path_returns) > 0:
            # if eval is too often, training may not even have collected a full
            # path
            train_returns = np.asarray(es_path_returns) / self.scale_reward
            self.last_statistics.update(create_stats_ordered_dict(
                'TrainingReturns', train_returns))

        for key, value in self.last_statistics.items():
            logger.record_tabular(key, value)

        return self.last_statistics

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            env=self.env,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )
