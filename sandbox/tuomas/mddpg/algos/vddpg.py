from sandbox.haoran.mddpg.misc.rllab_util import split_paths
from sandbox.haoran.mddpg.misc.data_processing import create_stats_ordered_dict
from sandbox.tuomas.mddpg.algos.online_algorithm import OnlineAlgorithm
from sandbox.rocky.tf.misc.tensor_utils import flatten_tensor_variables
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
from sandbox.tuomas.mddpg.misc.sampler import ParallelSampler

# for debugging
from sandbox.tuomas.mddpg.misc.sim_policy import rollout, rollout_alg

from rllab.misc.overrides import overrides
from rllab.misc import logger
from sandbox.tuomas.mddpg.misc import special
from rllab.envs.proxy_env import ProxyEnv
from rllab.core.serializable import Serializable

from collections import OrderedDict
import numpy as np
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gc

TARGET_PREFIX = "target_"
DUMMY_PREFIX = "dummy_"

def tf_shape(shape):
    """Converts a list of python and tf scalars tensors into a tf vector."""
    tf_shape_list = []
    for d in shape:
        if type(d) not in (np.int32, int, tf.Tensor):
            d = d.astype('int32')
        if type(d) == tf.Tensor:
            tf_shape_list.append(d)
        else:
            tf_shape_list.append(tf.constant(d))

    return tf.pack(tf_shape_list)


class VDDPG(OnlineAlgorithm, Serializable):
    """
    Variational DDPG with Stein Variational Gradient Descent using stochastic
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
            # Number of static particles (used only if actor_sparse_update=True)
            K_fixed=1,
            K_critic=99,
            Ks=None,  # Use different number of particles for different temps.
            q_prior=None,
            q_target_type="max",
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            Q_weight_decay=0.,
            alpha=1.,
            qf_extra_training=0,
            temperatures=None,
            train_critic=True,
            train_actor=True,
            actor_sparse_update=False,
            resume=False,
            n_eval_paths=2,
            svgd_target="action",
            plt_backend="TkAgg",
            target_action_dist="uniform",
            critic_train_frequency=1,
            actor_train_frequency=1,
            update_target_frequency=1,
            debug_mode=False,
            # evaluation
            axis3d=False,
            q_plot_settings=None,
            env_plot_settings=None,
            eval_kl_n_sample=10,
            eval_kl_n_sample_part=10,
            alpha_annealer=None,
            critic_subtract_value=False,
            critic_value_sampler='uniform',
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
        :param eval_kl_n_sample: (large values slow dow computation)
        :param eval_kl_n_sample_part: (large values slow dow computation)
        :return:
        """
        assert ((Ks is None and temperatures is None) or
                Ks.shape[0] == temperatures.shape[0])
        Serializable.quick_init(self, locals())
        self.kernel = kernel
        self.qf = qf
        self.q_prior = q_prior
        self.prior_coeff = 0.#1.
        self.K = K
        self.K_static = K_fixed
        self.Ks = Ks
        self.K_critic = K_critic  # Number of actions used to estimate the target.
        Da = env.action_space.flat_dim
        self.critic_proposal_sigma = np.ones((Da,))
        self.critic_proposal_mu = np.zeros((Da,))
        self.q_target_type = q_target_type
        self.critic_lr = qf_learning_rate
        self.critic_weight_decay = Q_weight_decay
        self.actor_learning_rate = policy_learning_rate
        self.alpha = alpha
        self.qf_extra_training = qf_extra_training
        self.really_train_critic = train_critic
        self.really_train_actor = train_actor
        self.actor_sparse_update = actor_sparse_update
        self.critic_subtract_value = critic_subtract_value
        self.critic_value_sampler = critic_value_sampler
        self.resume = resume
        self.temperatures = temperatures
        self.target_action_dist = target_action_dist
        self.critic_train_frequency = critic_train_frequency
        self.critic_train_counter = 0
        self.train_critic = self.really_train_critic # shall be modified later
        self.actor_train_frequency = actor_train_frequency
        self.actor_train_counter = 0
        self.train_actor = self.really_train_critic # shall be modified later
        self.update_target_frequency = update_target_frequency
        self.update_target_counter = 0
        self.update_target = self.train_critic
        self.debug_mode = debug_mode
        self.axis3d = axis3d
        self.q_plot_settings = q_plot_settings
        self.env_plot_settings = env_plot_settings



        self.true_env = env
        while isinstance(self.true_env, ProxyEnv):
            self.true_env = self.true_env._wrapped_env

        self.alpha_placeholder = tf.placeholder(tf.float32,
                                                shape=(),
                                                name='alpha')

        self.prior_coeff_placeholder = tf.placeholder(tf.float32,
                                                      shape=(),
                                                      name='prior_coeff')
        self.K_pl = tf.placeholder(tf.int32, shape=(), name='K')
        # # Number of particles for computing critic target.
        # self.K_critic_pl = tf.placeholder(tf.int32, shape=(), name='K')

        if q_target_type == 'soft':
            self.importance_weights_pl = tf.placeholder(
                tf.float32, shape=(None, self.K_critic, None),
                name='importance_weights'
            )

        self.svgd_target = svgd_target
        if svgd_target == "pre-action":
            assert policy.output_nonlinearity == tf.nn.tanh
            assert policy.output_scale == 1.

        assert train_actor or train_critic
        #assert isinstance(policy, StochasticNNPolicy)
        #assert isinstance(exploration_strategy, MNNStrategy)

        #if resume:
        #    qf_params = qf.get_param_values()
        #    policy_params = policy.get_param_values()
        super().__init__(env, policy, exploration_strategy, **kwargs)
        #if resume:
        #    qf.set_param_values(qf_params)
        #    policy.set_param_values(policy_params)

        self.eval_sampler = ParallelSampler(self)
        self.n_eval_paths = n_eval_paths
        plt.switch_backend(plt_backend)

        self.eval_kl_n_sample = eval_kl_n_sample
        self.eval_kl_n_sample_part = eval_kl_n_sample_part

        self.alpha_annealer = alpha_annealer

        self._init_figures()

    @overrides
    def _init_tensorflow_ops(self):

        # Useful dimensions.
        Da = self.env.action_space.flat_dim

        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())

        self.target_policy = self.policy.get_copy(
            scope_name=TARGET_PREFIX + self.policy.scope_name,
        )
        self.dummy_policy = self.policy.get_copy(
            scope_name=DUMMY_PREFIX + self.policy.scope_name,
        )

        if self.q_target_type == 'soft':
            # For soft target, we don't feed in the actions from the policy.
            self.target_qf = self.qf.get_copy(
                scope_name=TARGET_PREFIX + self.qf.scope_name,
            )
        else:
            self.target_qf = self.qf.get_copy(
                scope_name=TARGET_PREFIX + self.qf.scope_name,
                action_input=self.target_policy.output
            )

        # TH: It's a bit weird to set class attributes (kernel.kappa and
        # kernel.kappa_grads) outside the class. Could we do this somehow
        # differently?
        # Note: need to reshape policy output from N*K x Da to N x K x Da
        shape = tf_shape((-1, self.K_pl, Da))
        if self.svgd_target == "action":
            if self.actor_sparse_update:
                updated_actions = tf.reshape(self.policy.output, shape)
                self.kernel.kappa = self.kernel.get_asymmetric_kappa(
                    updated_actions
                )
                self.kernel.kappa_grads = (
                    self.kernel.get_asymmetric_kappa_grads(updated_actions)
                )

            else:
                actions_reshaped = tf.reshape(self.policy.output, shape)
                self.kernel.kappa = self.kernel.get_kappa(actions_reshaped)
                self.kernel.kappa_grads = self.kernel.get_kappa_grads(
                    actions_reshaped)

        elif self.svgd_target == "pre-action":
            if self.actor_sparse_update:
                updated_pre_actions_reshaped = tf.reshape(
                    self.policy.pre_output, shape
                )
                self.kernel.kappa = self.kernel.get_asymmetric_kappa(
                    updated_pre_actions_reshaped
                )
                self.kernel.kappa_grads =self.kernel.get_asymmetric_kappa_grads(
                    updated_pre_actions_reshaped
                )
            else:
                pre_actions_reshaped = tf.reshape(self.policy.pre_output, shape)
                self.kernel.kappa = self.kernel.get_kappa(pre_actions_reshaped)
                self.kernel.kappa_grads = self.kernel.get_kappa_grads(
                    pre_actions_reshaped)
        else:
            raise NotImplementedError

        self.kernel.sess = self.sess
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        if self.eval_policy:
            self.eval_policy.sess = self.sess
        self.target_policy.sess = self.sess
        self.dummy_policy.sess = self.sess

        self._init_ops()

        self.sess.run(tf.global_variables_initializer())

    def _init_ops(self):
        self._init_actor_ops()
        self._init_critic_ops()
        self._init_target_ops()

    def _init_actor_ops(self):
        """
        Note: critic is given as an argument so that we can have several critics

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
        if not self.train_actor:
            pass

        all_true_params = self.policy.get_params_internal()
        all_dummy_params = self.dummy_policy.get_params_internal()
        Da = self.env.action_space.flat_dim

        # TODO: not sure if this is needed
        self.critic_with_policy_input = self.qf.get_weight_tied_copy(
            action_input=self.policy.output,
            observation_input=self.policy.observations_placeholder,
        )
        if self.actor_sparse_update:
            if self.svgd_target == 'action':
                actions = tf.reshape(self.kernel.fixed_actions_pl,
                                     (-1, Da))
                self.critic_with_static_actions = self.qf.get_weight_tied_copy(
                    action_input=actions,
                    observation_input=None,
                    #observation_input=self.policy.observations_placeholder,
                )
            elif self.svgd_target == 'pre-action':
                #import pdb; pdb.set_trace()
                actions = tf.tanh(tf.reshape(self.kernel.fixed_actions_pl,
                                             (-1, Da)))
                self.critic_with_static_actions = self.qf.get_weight_tied_copy(
                    action_input=actions,
                    #observation_input=self.policy.observations_placeholder,
                    observation_input=None,
                )
            else:
                raise NotImplementedError

        if self.svgd_target == "action":
            if self.q_prior is not None and self.actor_sparse_update:
                raise NotImplementedError
            if self.q_prior is not None:
                self.prior_with_policy_input = self.q_prior.get_weight_tied_copy(
                    action_input=self.policy.output,
                    observation_input=self.policy.observations_placeholder,
                )
                p = self.prior_coeff_placeholder
                log_p = ((1.0 - p) * self.critic_with_policy_input.output
                         + p * self.prior_with_policy_input.output)
            elif self.actor_sparse_update:
                log_p = self.critic_with_static_actions.output
            else:
                log_p = self.critic_with_policy_input.output
            log_p = tf.squeeze(log_p)

            if self.actor_sparse_update:
                grad_log_p = tf.gradients(log_p,
                                          self.critic_with_static_actions.action_input)

                grad_log_p = tf.reshape(grad_log_p,
                                        tf_shape((-1, self.K_static, 1, Da)))
            else:
                grad_log_p = tf.gradients(log_p, self.policy.output)

                grad_log_p = tf.reshape(grad_log_p,
                                        tf_shape((-1, self.K_pl, 1, Da)))
            # N x K x 1 x Da

            kappa = tf.expand_dims(
                self.kernel.kappa,
                dim=3,
            )  # N x K x K x 1

            # grad w.r.t. left kernel input
            kappa_grads = self.kernel.kappa_grads  # N x K x K x Da

            # Stein Variational Gradient!
            action_grads = tf.reduce_mean(
                kappa * grad_log_p
                + self.alpha_placeholder * kappa_grads,
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
        elif self.svgd_target == "pre-action":
            if self.q_prior is not None and self.actor_sparse_update:
                raise NotImplementedError

            if self.q_prior is not None:
                self.prior_with_policy_input = self.q_prior.get_weight_tied_copy(
                    action_input=self.policy.output,
                    observation_input=self.policy.observations_placeholder,
                )
                p = self.prior_coeff_placeholder
                log_p_from_Q = ((1.0 - p) * self.critic_with_policy_input.output
                    + p * self.prior_with_policy_input.output)
            elif self.actor_sparse_update:
                log_p_from_Q = self.critic_with_static_actions.output  # N*K x 1
            else:
                log_p_from_Q = self.critic_with_policy_input.output # N*K x 1


            log_p_from_Q = tf.squeeze(log_p_from_Q) # N*K

            if self.actor_sparse_update:
                grad_log_p_from_Q = tf.gradients(
                    log_p_from_Q, 
                    self.kernel.fixed_actions_pl  # These are pre-actions
                )

                grad_log_p_from_tanh = - 2. * (
                    tf.tanh(self.kernel.fixed_actions_pl)
                )  # N*K x Da
            else:
                grad_log_p_from_Q = tf.gradients(log_p_from_Q, 
                                                 self.policy.pre_output)
                # N*K x Da
                grad_log_p_from_tanh = - 2. * self.policy.output # N*K x Da
                # d/dx(log(1-tanh^2(x))) = -2tanh(x)

            grad_log_p = (
                grad_log_p_from_Q +
                self.alpha_placeholder * grad_log_p_from_tanh
            )

            if self.actor_sparse_update:
                grad_log_p = tf.reshape(grad_log_p,
                                        tf_shape((-1, self.K_static, 1, Da)))
            else:
                grad_log_p = tf.reshape(grad_log_p,
                                        tf_shape((-1, self.K_pl, 1, Da)))
            # N x K x 1 x Da

            kappa = tf.expand_dims(
                self.kernel.kappa,
                dim=3,
            )  # N x K x K x 1

            # grad w.r.t. left kernel input
            kappa_grads = self.kernel.kappa_grads  # N x K x K x Da

            # Stein Variational Gradient!
            pre_action_grads = tf.reduce_mean(
                kappa * grad_log_p
                + self.alpha_placeholder * kappa_grads,
                reduction_indices=1,
            ) # N x K x Da

            # The first two dims needs to be flattened to correctly propagate the
            # gradients to the policy network.
            pre_action_grads = tf.reshape(pre_action_grads, (-1, Da))

            # Propagate the grads through the policy net.
            grads = tf.gradients(
                self.policy.pre_output,
                self.policy.get_params_internal(),
                grad_ys=pre_action_grads,
            )
        else:
            raise NotImplementedError

        self.actor_surrogate_loss = tf.reduce_mean(
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

    def _init_critic_ops(self):
        if not self.train_critic:
            return

        M = self.qf.dim if hasattr(self.qf, 'dim') else 1
        Da = self.env.action_space.flat_dim
        Do = self.env.observation_space.flat_dim

        if hasattr(self.target_qf, 'outputs'):
            q_next = self.target_qf.outputs
            q_curr = self.qf.outputs
        else:
            q_next = self.target_qf.output
            q_curr = self.qf.output

        # N x K x M
        q_next = tf.reshape(q_next, tf_shape((-1, self.K_critic, M)))
        q_curr = tf.reshape(q_curr, (-1, M))  # N x M

        # average across reward dims of max Q - min Q, where max and min are
        # over sampled actions
        # logging this allows us to know roughly how close are softmax and max
        self.q_next_avg_diff = tf.reduce_mean(
            tf.reduce_max(q_next, reduction_indices=1)\
            - tf.reduce_min(q_next, reduction_indices=1),
            reduction_indices=1,
        ) # N

        if self.q_target_type == 'mean':
            q_next = tf.reduce_mean(q_next, reduction_indices=1, name='q_next',
                                    keep_dims=False)  # N x M
        elif self.q_target_type == 'max':
            # TODO: This is actually wrong. Now the max of each critic might
            # be attained with a different actions. We should consistently
            # pick a single action and stick with that for all critics.
            q_next = tf.reduce_max(q_next, reduction_indices=1, name='q_next',
                                   keep_dims=False)  # N x M
        elif self.q_target_type == 'soft':
            # Note: q_next is actually soft V!
            q_next_max = tf.reduce_max(
                q_next,
                axis=1,
                keep_dims=True,
                name="q_next_max",
            ) # N x 1 x M
            exp_q_next_minus_max = tf.exp(q_next - q_next_max)  # N x K x M
            q_next = q_next_max[:,0,:] + tf.log(tf.reduce_mean(
                exp_q_next_minus_max / self.importance_weights_pl,
                axis=1,
                keep_dims=False,
                name="q_next",
            ))

        else:
            raise NotImplementedError
        # q_next: N x M

        assert_op = tf.assert_equal(
            tf.shape(self.rewards_placeholder), tf.shape(q_next)
        )

        with tf.control_dependencies([assert_op]):
            # TODO: Discount should be set independently for each critic.
            self.ys = (
                self.rewards_placeholder + (1 - self.terminals_placeholder) *
                self.discount * q_next
            )  # N x M

        if self.critic_subtract_value:
            # makes sure ADAM knows how to find the right gradient
            assert M == 1  # Don't know what happens if M is not 1

            if self.critic_value_sampler == 'policy':  # Use current policy for sampling.
                q_contrastive = tf.reshape(
                    self.critic_with_policy_input.output,
                    tf_shape((-1, self.K_pl, 1))
                )  # N x K x 1
                v = tf.reduce_mean(q_contrastive, axis=1)  # N x 1

                bellman_error = self.ys - q_curr  # N x 1
                advantage = q_curr - v  # N x 1
                self.critic_loss = tf.reduce_mean(tf.reduce_mean(
                    - advantage * tf.stop_gradient(bellman_error)
                ))
            elif self.critic_value_sampler == 'uniform':

                # TODO: hacky
                actions_pl = tf.placeholder(
                    tf.float32,
                    shape=[None, Da],
                    name='contrastive_actions',
                )
                observations_pl = tf.placeholder(
                    tf.float32,
                    shape=[None, Do],
                    name='contrastive_observations',
                )

                self.critic_contrastive = self.qf.get_weight_tied_copy(
                    action_input=actions_pl,
                    observation_input=observations_pl
                )  # N*K

                q = tf.reshape(self.critic_contrastive.output,
                               tf_shape((-1, self.K_critic)))  # N x K

                # N x 1
                contrastive_max = tf.reduce_max(q, axis=1, keep_dims=True)
                exp_q = tf.exp(q - contrastive_max)  # N x K

                scale = self.policy.output_scale

                w = 1. / np.power(2. * scale, Da)

                # N x 1
                mean_exp_q = w * tf.reduce_mean(exp_q, axis=1, keep_dims=True)
                v = tf.log(mean_exp_q)  # N x 1

                bellman_error = self.ys - q_curr  # N x 1
                advantage = q_curr - v  # N x 1
                self.critic_loss = tf.reduce_mean(tf.reduce_mean(
                    - advantage * tf.stop_gradient(bellman_error)
                ))
            else:
                raise NotImplementedError

        else:
            self.critic_loss = tf.reduce_mean(tf.reduce_mean(
                tf.square(self.ys - q_curr)
            ))

        self.critic_reg = tf.reduce_sum(
            tf.pack(
                [tf.nn.l2_loss(v)
                 for v in
                 self.qf.get_params_internal(only_regularizable=True)]
            ),
            name='weights_norm'
        )

        self.critic_total_loss = (
            self.critic_loss + self.critic_weight_decay * self.critic_reg)

        self.train_critic_op = tf.train.AdamOptimizer(self.critic_lr).minimize(
            self.critic_total_loss,
            var_list=self.qf.get_params_internal()
        )

    def _init_target_ops(self):

        if self.train_critic:
            # Set target policy
            actor_vars = self.policy.get_params_internal()
            target_actor_vars = self.target_policy.get_params_internal()
            assert len(actor_vars) == len(target_actor_vars)
            self.update_target_actor_op = [
                tf.assign(target, (self.tau * src + (1 - self.tau) * target))
                for target, src in zip(target_actor_vars, actor_vars)]

            # Set target Q-function
            critic_vars = self.qf.get_params_internal()
            target_critic_vars = self.target_qf.get_params_internal()
            self.update_target_critic_op = [
                tf.assign(target, self.tau * src + (1 - self.tau) * target)
                for target, src in zip(target_critic_vars, critic_vars)
            ]

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_qf.set_param_values(self.qf.get_param_values())
        self.target_policy.set_param_values(self.policy.get_param_values())
        self.dummy_policy.set_param_values(self.policy.get_param_values())

    @overrides
    def _get_training_ops(self):
        """ notice that the order of these ops are different from before """
        ops = []
        if self.train_actor:
            ops.append(self.train_actor_op)
            if self.debug_mode:
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
                    ops.append(
                        tf.Print(
                            self.tau,
                            [self.tau],
                            message="Update target actor with tau: "
                        )
                    )
        if self.train_critic:
            ops.append(self.train_critic_op)
            if self.debug_mode:
                ops.append(
                    tf.Print(
                        self.critic_total_loss,
                        [self.critic_total_loss],
                        message="Critic minibatch loss: ",
                    )
                )
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
        return ops

    def _get_finalize_ops(self):
        return [self.finalize_actor_op]

    #def _sample_temps(self, N):
    #    inds = np.random.randint(0, self.temperatures.shape[0], size=(N,))
    #    temps = self.temperatures[inds]
    #    return temps

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        # Note: each sample in a batch need to have the same K. That's why
        # also the temperature is same for all of them (we want associate
        # specific temperature values to specific Ks in order not to confuse
        # the network.
        N = obs.shape[0]
        if self.temperatures is not None:
            ind = np.random.randint(0, self.temperatures.shape[0])
            K = self.Ks[ind]
            temp = self.temperatures[[ind]]
            temp = self._replicate_rows(temp, N)
        else:
            temp = None
            K = self.K

        feeds = dict()
        if self.train_actor:
            if self.actor_sparse_update:
                # Update feed dict first with larger number of fixed particles
                feeds.update(self._actor_feed_dict_for(
                    self.critic_with_static_actions, obs, temp, self.K_static))
                feeds.update(
                    self.kernel.update(self,
                                       feeds,
                                       multiheaded=False,
                                       K=self.K_static)
                )

                # Then update the feeds again with smaller number of particles.
                feeds.update(self._actor_feed_dict_for(None, obs, temp, self.K))
            else:
                feeds.update(self._actor_feed_dict_for(
                    self.critic_with_policy_input, obs, temp, K)
                )
                feeds.update(
                    self.kernel.update(self, feeds, multiheaded=False,
                                       K=self.K)
                )

        if self.train_critic:
            feeds.update(self._critic_feed_dict(
                rewards, terminals, obs, actions, next_obs, temp, self.K_critic
            ))

        feeds[self.K_pl] = K

        return feeds

    def _actor_feed_dict_for(self, critic, obs, temp, K):
        # Note that we want K samples for each observation. Therefore we
        # first need to replicate the observations.
        obs = self._replicate_rows(obs, K)
        temp = self._replicate_rows(temp, K)

        # Make sure we're not giving extra arguments for policies not supporting
        # temperature input.
        actor_inputs = (obs,) if temp is None else (obs, temp)
        critic_inputs = (obs,) if temp is None else (obs, None, temp)

        feed = self.policy.get_feed_dict(*actor_inputs)
        #feed.update(self.critic_with_policy_input.get_feed_dict(*critic_inputs))
        if critic is not None:
            feed.update(critic.get_feed_dict(*critic_inputs))
        #feed.update(self.critic_with_static_actions.get_feed_dict(*critic_inputs))
        feed[self.alpha_placeholder] = self.alpha
        feed[self.prior_coeff_placeholder] = self.prior_coeff
        return feed

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                          temp, K):
        N = obs.shape[0]
        Da = self.env.action_space.flat_dim
        feed = {}
        # Again, we'll need to replicate next_obs.
        next_obs = self._replicate_rows(next_obs, K)

        if (self.critic_subtract_value and
                self.critic_value_sampler == 'uniform'):
            scale = self.policy.output_scale
            uniform_actions = np.random.uniform(
                low=-scale, high=scale, size=(N * self.K_critic, Da))

            feed.update(self.critic_contrastive.get_feed_dict(
                action=uniform_actions,
                obs=self._replicate_rows(obs, self.K_critic),
            ))

        # TODO: we should make the next temp really low (actually high since
        # it is inverse temperature)
        temp = self._replicate_rows(temp, K)

        target_policy_input = [next_obs]
        critic_input = [obs, actions]

        if self.q_target_type == 'soft':
            # We'll use the same actions for each sample (first dimension).

            if self.target_action_dist == "uniform":
                scale = self.policy.output_scale
                next_actions = np.random.uniform(
                    low=-scale, high=scale, size=(N, self.K_critic, Da))
                weights = np.power(2. * scale, -Da) * np.ones((N, self.K_critic))
                    # 1/p = volume of the action space
            elif self.target_action_dist == "gaussian":
                ## old Gaussian sampling: may accidentally get low prob samples
                next_actions = (np.random.randn(N, self.K_critic, Da)
                           * self.critic_proposal_sigma + self.critic_proposal_mu)
                weights = scipy.stats.multivariate_normal(
                    mean=self.critic_proposal_mu,
                    cov=self.critic_proposal_sigma**2,
                ).pdf(next_actions) #N*K_critic x Da
            # Be careful in using the "policy" option, as in the early stage of
            # training, the weights tend to be huge, because the actions are
            # concentrated on a "low-dimensional" manifold in the high-dim
            # aciton space
            elif self.target_action_dist == "policy":
                next_actions, info = self.policy.get_actions(
                    next_obs,
                    with_prob=True,
                )
                weights = info["prob"].reshape(N, self.K_critic)
            else:
                raise NotImplementedError

            next_actions = np.reshape(next_actions, (N*self.K_critic, Da))

            feed[self.importance_weights_pl] = weights[:, :, None]
            target_critic_input = [next_obs, next_actions]
        else:
            target_critic_input = [next_obs, None]

        if temp is not None:
            target_policy_input.append(temp)
            critic_input.append(temp)
            target_critic_input.append(temp)

        #curr_inputs = (obs, actions) if temp is None else (obs, actions, temp)
        #next_inputs = (next_obs,) if temp is None else (next_obs, temp)

        feed.update(self.target_policy.get_feed_dict(*target_policy_input))
        feed.update(self.qf.get_feed_dict(*critic_input))
        feed.update(self.target_qf.get_feed_dict(*target_critic_input))


        # Adjust rewards dims for backward compatibility.
        if rewards.ndim == 1:
            rewards = np.expand_dims(rewards, axis=1)

        feed.update({
            self.rewards_placeholder: rewards,
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
        })

        return feed

    def _replicate_rows(self, t, K):
        """Replicates each row in t K times."""
        if t is None:
            return t

        assert t.ndim == 2
        N = t.shape[0]

        t = np.expand_dims(t, axis=1)  # N x 1 x X
        t = np.tile(t, (1, K, 1))  # N x K x X
        t = np.reshape(t, (N * K, -1))  # N*K x Do

        return t

    def _init_figures(self):
        # Init environment figure.
        if self.env_plot_settings is not None:
            if "figsize" not in self.env_plot_settings.keys():
                figsize = (7,7)
            else:
                figsize = self.env_plot_settings['figsize']
            self._fig_env = plt.figure(
                figsize=figsize,
            )
            self._ax_env = self._fig_env.add_subplot(111)
            if hasattr(self.true_env, 'set_axis'):
                self.true_env.set_axis(self._ax_env)

            # List of holding line objects created by the environment
            self._env_lines = []
            # self._ax_env.set_xlim(self.env_plot_settings['xlim'])
            # self._ax_env.set_ylim(self.env_plot_settings['ylim'])

        # Init critic + actor figure.
        if self.q_plot_settings is not None:
            # Make sure the observations are given as np array.
            self.q_plot_settings['obs_lst'] = (
                np.array(self.q_plot_settings['obs_lst'])
            )
            n_states = len(self.q_plot_settings['obs_lst'])

            xsize = 5 * n_states
            ysize = 5

            self._fig_q = plt.figure(figsize=(xsize, ysize))

            self._ax_q_lst = []
            for i in range(n_states):
                ax = self._fig_q.add_subplot(100 + n_states * 10 + i + 1)
                self._ax_q_lst.append(ax)


    def compute_kl_entropy(self, observations, K, K_part):
        """
        K: number of particles to estimate log(q) / log(bar{p})
        K_part: number of particles to estimate the partition function
        """
        TINY = 1e-8
        N = observations.shape[0]
        Da = self.policy.action_dim
        Do = self.policy.observation_dim

        # compute the partition function
        scale = self.policy.output_scale
        actions_part = np.random.uniform(
            low=-scale, high=scale, size=(N * K_part, Da))
        weights = np.power(2. * scale, -Da) * np.ones((N, K_part))
        Qs_part = self.sess.run(
            self.qf.output,
            self.qf.get_feed_dict(
                obs=np.tile(observations, (1, K_part)).reshape(-1, Do),
                action=actions_part,
            )
        ).reshape(N, K_part)
        Qs_part_max = np.amax(Qs_part, axis=1, keepdims=True)
        logZs = Qs_part_max[:,0] + np.log(np.mean(
            np.exp(Qs_part - Qs_part_max) / weights,
            axis=1
        )) # (N,)
            # logZs are also the soft values

        # compute kl(q | bar{p})
        obs = np.tile(observations, (1, K)).reshape(-1, Do)
        actions, info = self.policy.get_actions(obs, with_prob=True)
        Qs = self.sess.run(
            self.qf.output,
            self.qf.get_feed_dict(obs, actions),
        ).reshape(N, K)

        qs = info["prob"].reshape(N, K)
        entropies = np.mean(-np.log(qs + TINY), axis=1)
        kl = np.mean(np.log(qs + TINY) - Qs, axis=1) + logZs

        entropy_bound = Da * np.log(2. * self.policy.output_scale)
            # the uniform policy has entropy - log (1/volume) = log(volume)
            # if output_scale = 1, the bound is Da * 0.7
        entropies_clipped = np.minimum(entropies, entropy_bound)

        return kl, entropies_clipped

    @overrides
    def evaluate(self, epoch, train_info):
        logger.log("Collecting samples for evaluation")
        paths = self.eval_sampler.obtain_samples(
            n_paths=self.n_eval_paths,
            max_path_length=self.max_path_length,
            policy=self.eval_policy
        )
        rewards, terminals, obs, actions, next_obs = split_paths(paths)

        # temperarily turn on these so that _update_feed_dict can work
        # this will not impact the training process
        self.train_actor = True
        self.train_critic = True
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
            kappa,  # N x K x K
            qf_next_avg_diff,
        ) = self.sess.run(
            [
                self.actor_surrogate_loss,
                self.critic_loss,
                self.policy.output,
                self.target_policy.output,
                self.qf.output,
                self.target_qf.output,
                self.ys,
                self.kernel.kappa,
                self.q_next_avg_diff,
            ],
            feed_dict=feed_dict)
        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount)
             for path in paths]
        )
        returns = np.asarray([sum(path["rewards"]) for path in paths])
        rewards = np.hstack([path["rewards"] for path in paths])
        Da = self.env.action_space.flat_dim
        policy_vars = np.mean(
            np.var(
                policy_outputs.reshape((-1, self.K, Da)),
                axis=1
            ), axis=1
        )
        kappa_sum = np.sum(kappa, axis=1).ravel()

        # Log statistics
        self.last_statistics.update(OrderedDict([
            ('Epoch', epoch),
            # ('PolicySurrogateLoss', policy_loss),
            ('CriticLoss', qf_loss),
            ('AverageDiscountedReturn', average_discounted_return),
            ('Alpha', self.alpha)
        ]))
        # self.last_statistics.update(create_stats_ordered_dict('Ys', ys))
        self.last_statistics.update(create_stats_ordered_dict('QfOutput',
                                                              qf_outputs))
        # self.last_statistics.update(create_stats_ordered_dict('TargetQfOutput',
        #                                                       target_qf_outputs))
        # self.last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        self.last_statistics.update(create_stats_ordered_dict('returns', returns))
        self.last_statistics.update(
           create_stats_ordered_dict('PolicyVars',policy_vars)
        )
        self.last_statistics.update(
            create_stats_ordered_dict('KappaSum',kappa_sum)
        )
        self.last_statistics.update(
            create_stats_ordered_dict('TargetQfAvgDiff',qf_next_avg_diff)
        )

        es_path_returns = train_info["es_path_returns"]
        if len(es_path_returns) == 0 and epoch == 0:
            es_path_returns = [0]
        if len(es_path_returns) > 0:
            # if eval is too often, training may not even have collected a full
            # path
            train_returns = np.asarray(es_path_returns) / self.scale_reward
            self.last_statistics.update(create_stats_ordered_dict(
                'TrainingReturns', train_returns))

        es_path_lengths = train_info["es_path_lengths"]
        if len(es_path_lengths) == 0 and epoch == 0:
            es_path_lengths = [0]
        if len(es_path_lengths) > 0:
            # if eval is too often, training may not even have collected a full
            # path
            self.last_statistics.update(create_stats_ordered_dict(
                'TrainingPathLengths', es_path_lengths))

        # log the entropy regularized objective
        # we should scale down the entropies if we have scaled up
        discounted_regularized_returns = []
        entropy_reward_ratios = []
        all_kls = []
        for path in paths:
            kls, entropies = self.compute_kl_entropy(
                    path["observations"],
                    K=self.eval_kl_n_sample,
                    K_part=self.eval_kl_n_sample_part,
                )
            all_kls = np.concatenate([all_kls, kls])
            entropy_bonuses = np.concatenate([[0], entropies[1:]])
            discounted_rewards = special.discount_return(
                path["rewards"], self.discount
            )
            discounted_entropies = special.discount_return(
                entropy_bonuses,  self.discount
            )
            discounted_regularized_returns.append(
                discounted_rewards + self.alpha / self.scale_reward * discounted_entropies
            )
            entropy_reward_ratios.append(
                self.alpha / self.scale_reward * discounted_entropies / discounted_rewards
            )
        self.last_statistics.update(create_stats_ordered_dict(
            'DiscRegReturn', discounted_regularized_returns))
        self.last_statistics.update(create_stats_ordered_dict(
            'EntropyRewardRatio', entropy_reward_ratios))
        self.last_statistics.update(create_stats_ordered_dict(
            'KL', all_kls))


        # Collect environment info.
        snapshot_dir = logger.get_snapshot_dir()
        env = self.env
        while isinstance(env, ProxyEnv):
            env = env._wrapped_env

        if hasattr(env, "log_stats"):
            env_stats = env.log_stats(self, epoch, paths)
            #env_stats = env.log_stats(epoch, paths)
            self.last_statistics.update(env_stats)

        if hasattr(env, 'plot_paths') and self.env_plot_settings is not None:
            img_file = os.path.join(snapshot_dir,
                                    'env_itr_%05d.png' % epoch)

            # Delete previous paths
            if self._env_lines is not None:
                [path.remove() for path in self._env_lines]

            #self._ax_env.clear()
            self._env_lines = env.plot_paths(paths, self._ax_env)
            # self._ax_env.set_xlim(self.env_plot_settings['xlim'])
            # self._ax_env.set_ylim(self.env_plot_settings['ylim'])

            plt.pause(0.001)
            plt.draw()

            self._fig_env.savefig(img_file, dpi=100)

        # Collect actor and critic info (save just plots)
        if hasattr(self.qf, 'plot') and self.q_plot_settings is not None:
            img_file = os.path.join(snapshot_dir,
                                    'q_itr_%05d.png' % epoch)

            [ax.clear() for ax in self._ax_q_lst]
            self.qf.plot(
                ax_lst=self._ax_q_lst,
                obs_lst=self.q_plot_settings['obs_lst'],
                action_dims=self.q_plot_settings['action_dims'],
                xlim=self.q_plot_settings['xlim'],
                ylim=self.q_plot_settings['ylim'],
            )

            self.policy.plot_samples(self._ax_q_lst,
                                     self.q_plot_settings['obs_lst'],
                                     self.K)

            plt.pause(0.001)
            plt.draw()

            self._fig_q.savefig(img_file, dpi=100)

        for key, value in self.last_statistics.items():
            logger.record_tabular(key, value)

        gc.collect()

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            # env=self.env,
            # policy=self.policy,
            # es=self.exploration_strategy,
            # qf=self.qf,
            # kernel=self.kernel,
            algo=self,
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

    def _do_training(self):
        self.train_critic = (np.mod(
            self.critic_train_counter,
            self.critic_train_frequency,
        ) == 0) and self.really_train_critic
        self.train_actor = (np.mod(
            self.actor_train_counter,
            self.actor_train_frequency,
        ) == 0) and self.really_train_actor
        self.update_target = (np.mod(
            self.update_target_counter,
            self.update_target_frequency,
        ) == 0) and self.really_train_critic

        super()._do_training()

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

    @overrides
    def update_training_settings(self, epoch):
        if self.scale_reward_annealer is not None:
            self.scale_reward = self.scale_reward_annealer.get_new_value(epoch)
        if self.alpha_annealer is not None:
            self.alpha = self.alpha_annealer.get_new_value(epoch)

## Use the following code to test whether the exp(Q-Qmax) code works
# import tensorflow as tf
# import numpy as np
#
# q_next = tf.placeholder(shape=(2,3,4),dtype=tf.float32)
# weights = tf.placeholder(shape=(2,3,1),dtype=tf.float32)
#
#
# q_next_max = tf.reduce_max(
#     q_next,
#     axis=1,
#     keep_dims=True,
#     name="q_next_max",
# ) # N x 1 x M
# exp_q_next_minus_max = tf.exp(q_next - q_next_max)  # N x K x M
# q_next1 = q_next_max[:,0,:] + tf.log(tf.reduce_mean(
#     exp_q_next_minus_max / weights,
#     axis=1,
#     keep_dims=False,
#     name="q_next1",
# ))
#
#
# ## inf errors?
# exp_q_next = tf.exp(q_next)  # N x K x M
# weighted_exp_q_samples = exp_q_next / weights
# # N x M
# q_next2 = tf.log(tf.reduce_mean(weighted_exp_q_samples, axis=1), name="q_next2")
#
# diff = q_next1 - q_next2
#
# with tf.Session() as sess:
#     feed = {
#         q_next: np.random.randn(2,3,4) * 1,
#         weights: np.random.uniform(low=1,high=2,size=(2,3,1)),
#     }
#     results = sess.run([q_next1,q_next2, diff], feed)
#     print(results)
