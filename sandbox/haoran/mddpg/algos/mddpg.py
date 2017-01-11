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


class MDDPG(OnlineAlgorithm):
    """
    Multiheaded DDPG with Stein Variational Gradient Descent
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
        assert isinstance(policy, MNNPolicy)
        assert policy.K == self.K
        assert isinstance(exploration_strategy, MNNStrategy)

        if resume:
            qf_params = qf.get_param_values()
            policy_params = policy.get_param_values()
        super().__init__(env, policy, exploration_strategy, **kwargs)
        if resume:
            qf.set_param_values(qf_params)
            policy.set_param_values(policy_params)


    @overrides
    def _init_tensorflow_ops(self):
        # Initialize variables for get_copy to work
        self.sess.run(tf.global_variables_initializer())
        self.target_policy = self.policy.get_copy(
            scope_name=TARGET_PREFIX + self.policy.scope_name,
        )
        self.dummy_policy = self.policy.get_copy(
            scope_name="dummy_" + self.policy.scope_name,
        )
        # The targets have shared weights, but differnet from current Q
        self.target_qf_list = []
        for k in range(self.K):
            if k == 0:
                target_qf = self.qf.get_copy(
                    scope_name=TARGET_PREFIX + self.qf.scope_name,
                    action_input=self.target_policy.heads[0]
                )
            else:
                target_qf = self.target_qf_list[0].get_weight_tied_copy(
                    action_input=self.target_policy.heads[k]
                )
            self.target_qf_list.append(target_qf)
        self.target_qf_outputs = tf.pack(
            [target_qf.output for target_qf in self.target_qf_list],
            axis=1,
        ) # N x K
        # TH: It's a bit weird to set class attributes (kernel.kappa and
        # kernel.kappa_grads) outside the class. Could we do this somehow
        # differently?
        self.kernel.kappa = self.kernel.get_kappa(self.policy.output)
        self.kernel.kappa_grads = self.kernel.get_kappa_grads(
            self.policy.output)

        self.kernel.sess = self.sess
        self.qf.sess = self.sess
        self.policy.sess = self.sess
        for target_qf in self.target_qf_list:
            target_qf.sess = self.sess
        self.target_policy.sess = self.sess
        self.dummy_policy.sess = self.sess
        # if not self.only_train_actor:
        self._init_critic_ops()
        if not self.only_train_critic:
            self._init_actor_ops()
        self._init_target_ops()
        self.sess.run(tf.global_variables_initializer())

    def _init_critic_ops(self):
        self.all_target_qs = [
            (self.rewards_placeholder +
                (1. - self.terminals_placeholder) *
                self.discount * qf.output)
            for qf in self.target_qf_list
        ] # K x N
        if self.q_target_type == "max":
            self.ys = tf.reduce_max(
                self.all_target_qs,
                reduction_indices=0,
                name="targets"
            )
        elif self.q_target_type == "mean":
            self.ys = tf.reduce_mean(
                self.all_target_qs,
                reduction_indices=0,
                name="targets"
            )
        elif self.q_target_type == "none":
            # for debugging, do not update the Q value
            self.ys = self.qf.output
        else:
            raise NotImplementedError

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
        # hack: avoid updating Q
        if self.q_target_type == "none":
            self.train_critic_op = tf.constant(1.)
        else:
            if not self.only_train_actor:
                self.train_critic_op = tf.train.AdamOptimizer(
                    self.critic_learning_rate).minimize(
                    self.critic_total_loss,
                    var_list=self.qf.get_params_internal())
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

        # Should use current policy and Q, as targets only update slowly
        self.critics_with_action_input = [
            self.qf.get_weight_tied_copy(self.policy.heads[j])
            for j in range(self.K)
        ]
        tmp = tf.expand_dims(
            tf.pack([
                tf.gradients(
                    self.critics_with_action_input[j].output,
                    self.policy.heads[j],
                )[0]
                for j in range(self.K)
            ]),
            dim=1,
        )

        qf_grads = tf.transpose(
            # here the dimensions are (j,k,N,d)
            tmp,
            [2,0,1,3]
            # then it becomse (N,j,k,d)
        ) # \nabla_a Q(s,a)
        kappa = tf.expand_dims(
            self.kernel.kappa,
            dim=3,
        )
        # grad w.r.t. left kernel inut
        kappa_grads = self.kernel.kappa_grads

        # sum (not avg) over j
        # using sum ensures that the gradient scale is close to DDPG
        # since kappa(x,x) = 1, but we may need to use different learning rates
        # for different numbers of heads
        # TH: Changed this from sum to average for easier debugging.
        #action_grads = tf.reduce_sum(
        action_grads = tf.reduce_mean(
            kappa * qf_grads + self.alpha * kappa_grads,
            reduction_indices=1,
        ) # (N,k,d)
        # ---------------------------------------------------------------

        # propagate action grads to NN parameter grads
        # this part computes sum_{i,k,l} (action_grads[i,k,l] *
        #   d(action[i,k,l]) / d theta), where
        # i: sample index
        # k: head index
        # l: action dimension index
        # actually grads is not a 1D vector, but a list of tensors corresponding
        # to weights and biases of different layers; but
        #   np.concatenate([g.ravel() for g in grads])
        # gives you the flat gradient


        grads = tf.gradients(
            self.policy.output,
            all_true_params,
            grad_ys=action_grads,
        )

        # In case you doubt, flat grads is essentially the same as below
        # flat_grads = np.zeros_like(self.policy.get_param_values())
        # for i in range(N):
        #     for k in range(K):
        #         for l in range(d):
        #             low_grad = (
        #                 np.concatenate([
        #                     g.ravel() for g in
        #                     self.sess.run(
        #                         tf.gradients(
        #                             self.policy.output[i,k,l],
        #                             self.policy.get_params_internal(),
        #                         ),
        #                         feed_dict
        #                     )
        #                 ])
        #             ) # length L
        #             high_grad = agrads[i,k,l]
        #             full_grad = low_grad * high_grad
        #             flat_grads += full_grad

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

        # TH: Assign op needs to be run AFTER optimizer. To guarantee the right
        # order, they need to be run with separate tf.run calls.
        #self.train_actor_op += [
        self.finalize_actor_op = [
            tf.assign(true_param, dummy_param)
            for true_param, dummy_param in zip(
                all_true_params,
                all_dummy_params,
            )
        ]

    def _init_target_ops(self):
        if not self.only_train_critic:
            actor_vars = self.policy.get_params_internal()
            target_actor_vars = self.target_policy.get_params_internal()
            assert len(actor_vars) == len(target_actor_vars)
            self.update_target_actor_op = [
                tf.assign(target, (self.tau * src + (1 - self.tau) * target))
                for target, src in zip(target_actor_vars, actor_vars)]

        if not self.only_train_actor:
            # Since target Q functions share weights, only update one of them
            critic_vars = self.qf.get_params_internal()
            target_critic_vars = self.target_qf_list[0].get_params_internal()
            assert len(critic_vars) == len(target_critic_vars)
            self.update_target_critic_op = [
                tf.assign(target, (self.tau * src + (1 - self.tau) * target))
                for target, src in zip(target_critic_vars, critic_vars)
            ]

    @overrides
    def _init_training(self):
        super()._init_training()
        self.target_qf_list[0].set_param_values(self.qf.get_param_values())
        self.target_policy.set_param_values(self.policy.get_param_values())
        self.dummy_policy.set_param_values(self.policy.get_param_values())

    @overrides
    def _get_training_ops(self):
        if self.only_train_critic:
            train_ops = [
                self.train_critic_op,
                self.update_target_critic_op,
            ]
        elif self.only_train_actor:
            train_ops = [
                self.train_actor_op,
                self.update_target_actor_op,
            ]
        else:
            basic_ops = [
                self.train_actor_op,
                self.train_critic_op,
                self.update_target_critic_op,
                self.update_target_actor_op,
            ]
            extra_ops = [
                self.train_critic_op,
                self.update_target_critic_op,
            ] * self.qf_extra_training
            train_ops = basic_ops + extra_ops
        return train_ops

    def _get_finalize_ops(self):
        return [self.finalize_actor_op]

    @overrides
    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        critic_feed = self._critic_feed_dict(rewards,
                                             terminals,
                                             obs,
                                             actions,
                                             next_obs)
        actor_feed = self._actor_feed_dict(obs)
        kernel_feed = self.kernel.update(self, actor_feed)
        return {**critic_feed, **actor_feed, **kernel_feed}

    def _critic_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        feed = {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observations_placeholder: obs,
            self.qf.actions_placeholder: actions,
            self.target_policy.observations_placeholder: next_obs,
        }
        for qf in self.critics_with_action_input:
            feed[qf.observations_placeholder] = obs
        for target_qf in self.target_qf_list:
            feed[target_qf.observations_placeholder] = next_obs
        return feed

    def _actor_feed_dict(self, obs):
        return {
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
            kappa, # N x K x K
        ) = self.sess.run(
            [
                self.actor_surrogate_loss,
                self.critic_loss,
                self.policy.output,
                self.target_policy.output,
                self.qf.output,
                self.target_qf_outputs,
                self.ys,
                self.kernel.kappa,
            ],
            feed_dict=feed_dict)
        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount)
             for path in paths]
        )
        returns = np.asarray([sum(path["rewards"]) for path in paths])
        rewards = np.hstack([path["rewards"] for path in paths])
        policy_vars = np.mean(np.var(policy_outputs,axis=1),axis=1)
        kappa_sum = np.sum(kappa,axis=1).ravel()

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
        self.last_statistics.update(
            create_stats_ordered_dict('PolicyVars',policy_vars)
        )
        self.last_statistics.update(
            create_stats_ordered_dict('KappaSum',kappa_sum)
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

        true_env = self.env
        while isinstance(true_env,ProxyEnv):
            true_env = true_env._wrapped_env
        env_stats = true_env.log_stats(epoch, paths)
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
            kernel=self.kernel,
        )
