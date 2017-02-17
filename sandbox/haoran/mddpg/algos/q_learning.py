"""
:author: Haoran Tang
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

from rllab.policies.base import Policy
class BoltzmannPolicy(Policy):
    """
    pi(a|s) \propto_a exp(Q(s,a))
    actions sampled from a uniform grid
    """
    def __init__(
        self,
        scope_name,
        qf,
        env_spec,
        grid_size,
        alpha,
    ):
        Serializable.quick_init(self, locals())
        super(BoltzmannPolicy,self).__init__(env_spec)
        self.qf = qf
        self.grid_size = grid_size
        self.alpha = alpha

        lbs, ubs = env_spec.action_space.bounds
        xx_list = [
            np.arange(lb, ub, grid_size)
            for lb, ub in zip(lbs, ubs)
        ]

        X_list = np.meshgrid(*xx_list)
        self.action_candidates = np.vstack([X.ravel() for X in X_list]).transpose()
        self.n_action = self.action_candidates.shape[0]


    def get_action(self, observation):
        copied_obs = np.vstack([observation for a in range(self.n_action)])
        feed_dict = {
            self.qf.observations_placeholder: copied_obs,
            self.qf.actions_placeholder: self.action_candidates
        }
        all_qs = self.sess.run(self.qf.output, feed_dict)
        un_probs = np.exp(self.alpha * all_qs)
        probs = un_probs / np.sum(un_probs)
        action_index = special.weighted_sample(probs, range(self.n_action))
        action = self.action_candidates[action_index]
        return action, {}



class QLearning(OnlineAlgorithm):
    """
    A simple Q-learning algorithm for continuous control.
    The action space is discretized and the induced policy is a Boltzmann
        distribution over actions.
    This algo may give us insights about DDPG and MDDPG.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            qf,
            qf_learning_rate=1e-3,
            Q_weight_decay=0.,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
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
        self.sess.run(tf.initialize_all_variables())
        self.target_qf = self.qf.get_copy(
            scope_name=TARGET_PREFIX + self.qf.scope_name,
            action_input=
        )
        self.qf.sess = self.sess
        self.target_qf.sess = self.sess
        self._init_critic_ops()
        self._init_target_ops()
        self.sess.run(tf.initialize_all_variables())

    def _init_critic_ops(self):
        all_target_qs = tf.reshape(
            self.target_qf.output, # (N x |A|,)
            (-1,self.policy.n_action)
        ) # N x |A|
        unnormalized_target_probs = tf.exp(self.policy.alpha * all_target_qs) # N x |A|
        normalization_factors = tf.reduce_sum(
            unnormalized_target_probs,
            reduction_indices=1,
            keep_dims=True,
        ) # N x |A|
        target_probs = unnormalized_target_probs / normalization_factors # N x |A|
        target_vs = tf.reduce_sum(
            target_probs * all_target_qs,
            reduction_indices=1,
        ) # N

        self.ys = (
            self.rewards_placeholder +
            (1. - self.terminals_placeholder) *
            self.discount * target_vs)
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
        critic_vars = self.qf.get_params_internal()
        target_critic_vars = self.target_qf.get_params_internal()
        assert len(critic_vars) == len(target_critic_vars)

        self.update_target_critic_op = [
            tf.assign(target, (self.tau * src + (1 - self.tau) * target))
            for target, src in zip(target_critic_vars, critic_vars)]

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_qf.set_param_values(self.qf.get_param_values())

    @overrides
    def _get_training_ops(self):
        return [
            self.train_critic_op,
            self.update_target_critic_op,
        ]

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        critic_feed = self._critic_feed_dict(rewards,
                                             terminals,
                                             obs,
                                             actions,
                                             next_obs)
        return {**critic_feed, **actor_feed}

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        n_sample = len(rewards)

        copied_next_obs = []
        for o in next_obs:
            copied_next_obs += [o] * self.policy.n_action
        copied_next_obs = np.array(copied_next_obs)

        copied_next_actions = np.concatenate([
            self.policy.action_candidates
            for i in range(n_sample)
        ], axis=0)
        import pdb; pdb.set_trace()


        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: actions,
            self.target_qf.observations_placeholder: copied_next_obs,
        }

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
