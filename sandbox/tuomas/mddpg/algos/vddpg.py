from sandbox.haoran.mddpg.algos.online_algorithm import OnlineAlgorithm
from sandbox.rocky.tf.misc.tensor_utils import flatten_tensor_variables
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy

from rllab.misc.overrides import overrides

import numpy as np
import tensorflow as tf

TARGET_PREFIX = "target_"


class VDDPG(OnlineAlgorithm):
    """
    Multiheaded DDPG with Stein Variational Gradient Descent using stochastic
    net.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy,
            kernel,
            qf,
            K,
            q_target_type="max",
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            Q_weight_decay=0.,
            alpha=1.,
            qf_extra_training=0,
            only_train_critic=False,
            only_train_actor=False,
            resume=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: a multiheaded policy
        :param kernel: specifies discrepancy between heads
        :param qf: QFunctions that is Serializable
        :param K: number of policies
        :param q_target_type: how to aggregate targets from multiple heads
        :param qf_learning_rate: Learning rate of the critic
        :param policy_learning_rate: Learning rate of the actor
        :param Q_weight_decay: How much to decay the weights for Q
        :return:
        """
        self.kernel = kernel
        self.qf = qf
        self.K = K
        self.q_target_type = q_target_type
        self.critic_learning_rate = qf_learning_rate
        self.actor_learning_rate = policy_learning_rate
        self.Q_weight_decay = Q_weight_decay
        self.alpha = alpha
        self.qf_extra_training = qf_extra_training
        self.only_train_critic = only_train_critic
        self.only_train_actor = only_train_actor
        self.resume = resume

        assert not (only_train_actor and only_train_critic)
        assert isinstance(policy, StochasticNNPolicy)
        #assert isinstance(exploration_strategy, MNNStrategy)

        if resume:
            qf_params = qf.get_param_values()
            policy_params = policy.get_param_values()
        super().__init__(env, policy, exploration_strategy, **kwargs)
        if resume:
            qf.set_param_values(qf_params)
            policy.set_param_values(policy_params)


    @overrides
    def _init_tensorflow_ops(self):

        # Useful dimensions.
        Da = self.env.action_space.flat_dim
        K = self.K

        # Initialize variables for get_copy to work
        self.sess.run(tf.initialize_all_variables())
        self.target_policy = self.policy.get_copy(
            scope_name=TARGET_PREFIX + self.policy.scope_name,
        )
        self.dummy_policy = self.policy.get_copy(
            scope_name="dummy_" + self.policy.scope_name,
        )

        # TH: It's a bit weird to set class attributes (kernel.kappa and
        # kernel.kappa_grads) outside the class. Could we do this somehow
        # differently?
        # Note: need to reshape policy output from N*K x Da to N x K x Da
        actions_reshaped = tf.reshape(self.policy.output, (-1, K, Da))
        self.kernel.kappa = self.kernel.get_kappa(actions_reshaped)
        self.kernel.kappa_grads = self.kernel.get_kappa_grads(
            actions_reshaped)

        self.kernel.sess = self.sess
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        self.target_policy.sess = self.sess
        self.dummy_policy.sess = self.sess
        # if not self.only_train_actor:
        self._init_actor_ops()
        self.sess.run(tf.initialize_all_variables())

    def _init_actor_ops(self):
        """
        SVGD
        For easy coding, we can run a session first to update the kernel.
            But it means we need to compute the actor outputs twice. A
            benefit is the kernel sizes can be easily adapted. Otherwise
            tensorflow may differentiate through the kernel size as well.
        An alternative is to manually compute the gradient, but within
            one session.
        A third way is to feed gradients w.r.t. actions to tf.gradients by
            specifying grad_ys.
        Need to write a test case.
        """
        all_true_params = self.policy.get_params_internal()
        all_dummy_params = self.dummy_policy.get_params_internal()
        Da = self.env.action_space.flat_dim

        critic_with_policy_input = self.qf.get_weight_tied_copy(
            self.policy.output, self.qf.observations_placeholder)
        grad_qf = tf.gradients(critic_with_policy_input.output,
                               self.policy.output)  # N*K x 1 x Da
        grad_qf = tf.reshape(grad_qf, [-1, self.K, 1, Da])  # N x K x 1 x Da

        kappa = tf.expand_dims(
            self.kernel.kappa,
            dim=3,
        )  # N x K x K x 1

        # grad w.r.t. left kernel input
        kappa_grads = self.kernel.kappa_grads  # N x K x K x Da

        # Stein Variational Gradient!
        action_grads = tf.reduce_mean(
            kappa * grad_qf + self.alpha * kappa_grads,
            reduction_indices=1,
        ) # N x K x Da
        # The first two dims needs to be flattened to correctly propagate the
        # gradients to the policy network.
        action_grads = tf.reshape(action_grads, (-1, Da))

        # Propagate the grads through the policy net.
        grads = tf.gradients(
            self.policy.output,
            self.policy.get_params_internal(),
            grad_ys=action_grads,
        )

        self.actor_surrogate_loss = tf.reduce_sum(
            - flatten_tensor_variables(all_dummy_params) *
            flatten_tensor_variables(grads)
        )

        self.train_actor_op = [
            tf.train.AdamOptimizer(
                self.actor_learning_rate).minimize(
                self.actor_surrogate_loss,
                var_list=all_dummy_params)
        ]

        self.finalize_actor_op = [
            tf.assign(true_param, dummy_param)
            for true_param, dummy_param in zip(
                all_true_params,
                all_dummy_params,
            )
        ]

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_policy.set_param_values(self.policy.get_param_values())
        self.dummy_policy.set_param_values(self.policy.get_param_values())

    @overrides
    def _get_training_ops(self):
        if self.only_train_critic:
            raise NotImplementedError
        elif self.only_train_actor:
            train_ops = [
                self.train_actor_op,
            ]
        else:
            raise NotImplementedError
        return train_ops

    def _get_finalize_ops(self):
        return [self.finalize_actor_op]

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        actor_feed = self._actor_feed_dict(obs)
        kernel_feed = self.kernel.update(self, actor_feed, multiheaded=False,
                                         K=self.K)
        return {**actor_feed, **kernel_feed}

    def _actor_feed_dict(self, obs):
        # Note that we want K samples for each observation. Therefore we
        # first need to replicate
        Do = self.env.observation_space.flat_dim

        obs = np.expand_dims(obs, axis=1)  # N x 1 x Do
        obs = np.tile(obs, (1, self.K, 1))  # N x K x Do
        obs = np.reshape(obs, (-1, Do))  # N*K x Do

        feed = self.policy.get_feed_dict(obs)
        feed[self.qf.observations_placeholder] = obs
        return feed
