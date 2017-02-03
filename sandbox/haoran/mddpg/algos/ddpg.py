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

from sandbox.haoran.mddpg.misc.simple_replay_pool import SimpleReplayPool
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable

TARGET_PREFIX = "target_"


class DDPG(OnlineAlgorithm, Serializable):
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
            plt_backend="MacOSX",
            critic_train_frequency=1,
            actor_train_frequency=1,
            update_target_frequency=1,
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
        Serializable.quick_init(self, locals())
        self.qf = qf
        self.critic_learning_rate = qf_learning_rate
        self.actor_learning_rate = policy_learning_rate
        self.Q_weight_decay = Q_weight_decay
        self.plt_backend = plt_backend
        plt.switch_backend(plt_backend)
        self.critic_train_frequency = critic_train_frequency
        self.critic_train_counter = 0
        self.train_critic = True # shall be modified later
        self.actor_train_frequency = actor_train_frequency
        self.actor_train_counter = 0
        self.train_actor = True # shall be modified later
        self.update_target_frequency = update_target_frequency
        self.update_target_counter = 0
        self.update_target = True

        super().__init__(env, policy, exploration_strategy, **kwargs)

    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
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
        self.sess.run(tf.global_variables_initializer())

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
            action_input=self.policy.output,
            observation_input=self.policy.observations_placeholder
        )
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
        return []

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_qf.set_param_values(self.qf.get_param_values())
        self.target_policy.set_param_values(self.policy.get_param_values())

    @overrides
    def _get_training_ops(self):
        # return [
        #     self.train_actor_op,
        #     self.train_critic_op,
        #     self.update_target_critic_op,
        #     self.update_target_actor_op,
        # ]

        # notice that the order of these ops are different from above
        ops = []
        if self.train_actor:
            ops.append(self.train_actor_op)
            if self.update_target:
                ops.append(self.update_target_actor_op)
        if self.train_critic:
            ops.append(self.train_critic_op)
            # ops.append(
            #     tf.Print(
            #         self.critic_total_loss,
            #         [self.critic_total_loss],
            #         message="Critic minibatch loss: ",
            #     )
            # )
            if self.update_target:
                ops.append(self.update_target_critic_op)
        return ops

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        critic_feed = self._critic_feed_dict(rewards,
                                             terminals,
                                             obs,
                                             actions,
                                             next_obs)
        actor_feed = self._actor_feed_dict(obs)
        feed = {}
        if self.train_critic:
            feed.update(critic_feed)
        if self.train_actor:
            feed.update(actor_feed)
        return feed

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.policy.observations_placeholder: obs,
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
    def _start_worker(self):
        self.eval_sampler.start_worker()

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

        # Create figure for plotting the environment.
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)

        true_env = get_true_env(self.env)
        if hasattr(true_env, "log_stats"):
            env_stats = true_env.log_stats(self, epoch, paths, ax)
            self.last_statistics.update(env_stats)

        # Close and save figs.
        snapshot_dir = logger.get_snapshot_dir()
        if snapshot_dir is None:
            snapshot_dir = '/tmp/ddpg/'
            os.system('mkdir -p %s'%(snapshot_dir))
        img_file = os.path.join(snapshot_dir, 'itr_%d_test_paths.png' % epoch)

        plt.draw()
        plt.pause(0.001)

        plt.savefig(img_file, dpi=100)
        plt.cla()
        plt.close('all')

        for key, value in self.last_statistics.items():
            logger.record_tabular(key, value)

        return self.last_statistics

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            # env=self.env,
            # policy=self.policy,
            # es=self.exploration_strategy,
            # qf=self.qf,
            algo=self,
        )

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

        minibatch = self.pool.random_batch(self.batch_size)
        sampled_obs = minibatch['observations']
        sampled_terminals = minibatch['terminals']
        sampled_actions = minibatch['actions']
        sampled_rewards = minibatch['rewards'][:,0] # assume single reward
        sampled_next_obs = minibatch['next_observations']

        feed_dict = self._update_feed_dict(sampled_rewards,
                                           sampled_terminals,
                                           sampled_obs,
                                           sampled_actions,
                                           sampled_next_obs)


        # TH: First train, then finalize. This can be suboptimal.
        self.sess.run(self._get_training_ops(), feed_dict=feed_dict)
        self.sess.run(self._get_finalize_ops(), feed_dict=feed_dict)

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

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d.update({
            "policy_params": self.policy.get_param_values(),
            "qf_params": self.qf.get_param_values(),
        })
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.qf.set_param_values(d["qf_params"])
        self.policy.set_param_values(d["policy_params"])
