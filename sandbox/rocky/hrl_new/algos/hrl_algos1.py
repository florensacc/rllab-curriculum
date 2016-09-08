


from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.envs.base import EnvSpec
import sandbox.rocky.tf.core.layers as L
from rllab.misc import logger

import tensorflow as tf
import numpy as np


class PolicyImitationTrainer(object):
    def __init__(self, optimizer=None, max_epochs=10):
        if optimizer is None:
            optimizer = FirstOrderOptimizer(max_epochs=max_epochs, verbose=True)
        self.optimizer = optimizer

    def update_opt(self, policy, obs_var, state_info_vars_list, old_dist_info_vars_list):
        """
        :type policy: StochasticPolicy
        """

        state_info_vars = dict(list(zip(policy.state_info_keys, state_info_vars_list)))
        old_dist_info_vars = dict(list(zip(policy.distribution.dist_info_keys, old_dist_info_vars_list)))

        dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
        kl_sym = policy.distribution.kl_sym(old_dist_info_vars, dist_info_vars)
        # Objective is to simply minimize the KL
        loss = tf.reduce_mean(kl_sym)
        self.optimizer.update_opt(loss, policy, [obs_var] + state_info_vars_list + old_dist_info_vars_list)

    def train(self, inputs):
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)


class PolicyImprovementTrainer(object):
    def __init__(self, optimizer=None):
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer()
        self.optimizer = optimizer

    def update_opt(self, policy, step_size, obs_var, action_var, advantage_var, state_info_vars_list,
                   old_dist_info_vars_list):
        """
        :type policy: StochasticPolicy
        """
        state_info_vars = dict(list(zip(policy.state_info_keys, state_info_vars_list)))
        old_dist_info_vars = dict(list(zip(policy.distribution.dist_info_keys, old_dist_info_vars_list)))

        dist = policy.distribution

        dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        mean_kl = tf.reduce_mean(kl)
        surr_loss = - tf.reduce_mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list

        self.optimizer.update_opt(
            loss=surr_loss,
            target=policy,
            leq_constraint=(mean_kl, step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

    def train(self, inputs):
        loss_before = self.optimizer.loss(inputs)
        mean_kl_before = self.optimizer.constraint_val(inputs)
        self.optimizer.optimize(inputs)
        mean_kl = self.optimizer.constraint_val(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)


class BonusImitationTrainer(object):
    def __init__(self, bonus_evaluator, bonus_coeff, optimizer=None, max_epochs=10):
        if optimizer is None:
            optimizer = FirstOrderOptimizer(max_epochs=max_epochs, verbose=True)
        self.optimizer = optimizer
        self.bonus_evaluator = bonus_evaluator
        self.bonus_coeff = bonus_coeff
        self.loss_coeff_var = tf.Variable(initial_value=1., name="loss_coeff", dtype=tf.float32)

    def update_opt(self, policy, obs_var, state_info_vars_list, old_dist_info_vars_list):
        """
        :type policy: StochasticPolicy
        """

        state_info_vars = dict(list(zip(policy.state_info_keys, state_info_vars_list)))
        old_dist_info_vars = dict(list(zip(policy.distribution.dist_info_keys, old_dist_info_vars_list)))

        dist_info_vars = policy.dist_info_sym(obs_var, state_info_vars)
        kl_sym = policy.distribution.kl_sym(old_dist_info_vars, dist_info_vars)

        bonus_sym = self.bonus_evaluator.bonus_sym(obs_var, state_info_vars)

        # Objective is to simply minimize the KL
        loss = tf.reduce_mean(kl_sym)

        joint_loss = self.loss_coeff_var * loss - bonus_sym

        self.optimizer.update_opt(joint_loss, policy, [obs_var] + state_info_vars_list + old_dist_info_vars_list)
        self.f_loss = tensor_utils.compile_function(
            inputs=[obs_var] + state_info_vars_list + old_dist_info_vars_list,
            outputs=[loss, bonus_sym, joint_loss]
        )

    def train(self, inputs):
        loss_before, bonus_before, joint_loss_before = self.f_loss(*inputs)
        # initialize bonus coefficient to be the largest
        sess = tf.get_default_session()
        sess.run(tf.assign(self.loss_coeff_var, 1.0))

        def opt_callback(*args, **kwargs):
            cur_coeff = sess.run(self.loss_coeff_var)
            upscale_coeff = 10.
            logger.log("Growing loss_coeff from %f to %f" % (cur_coeff, cur_coeff * upscale_coeff))
            sess.run(tf.assign(self.loss_coeff_var, cur_coeff * upscale_coeff))

        self.optimizer.optimize(inputs, callback=opt_callback)
        sess.run(tf.assign(self.loss_coeff_var, 1.0))
        loss_after, bonus_after, joint_loss_after = self.f_loss(*inputs)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('BonusBefore', bonus_before)
        logger.record_tabular('BonusAfter', bonus_after)
        logger.record_tabular('JointLossBefore', joint_loss_before)
        logger.record_tabular('JointLossAfter', joint_loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        logger.record_tabular('dBonus', bonus_after - bonus_before)
        logger.record_tabular('dJointLoss', joint_loss_before - joint_loss_after)


class BonusEvaluator(object):
    def __init__(self, env_spec, policy):
        """
        :type env_spec: EnvSpec
        :type policy: FixedClockPolicy
        :return:
        """
        self.env_spec = env_spec
        self.policy = policy

    # def p_g_given_z_sym(self, obs_var, state_info_vars):
    #     subgoal_obs = state_info_vars["subgoal_obs"]
    #     subgoal_probs = L.get_output(self.policy.l_subgoal_prob, {self.policy.l_obs: subgoal_obs})
    #     bottleneck_probs = L.get_output(self.policy.l_bottleneck_prob, {self.policy.l_obs: obs_var})
    #     N = tf.shape(obs_var)[0]
    #     p_z = tf.reduce_mean(bottleneck_probs, reduction_indices=0)
    #     p_g_z = tf.matmul(tf.transpose(subgoal_probs), bottleneck_probs) / tf.cast(N, tf.float32)
    #     p_g_given_z = p_g_z / tf.expand_dims(p_z, 0)
    #     return p_g_given_z, p_g_z, p_z

    def mi_a_g_given_z_sym(self, obs_var, state_info_vars):
        action_dim = self.policy.action_dim

        # compute p(g|z), p(g,z), and p(z), which are, respectively:
        # p(g|z): a matrix of dimension |G|*|Z|
        # p(g,z): a matrix of dimension |G|*|Z|
        # p(z): a vector of dimension |Z|

        subgoal_obs = state_info_vars["subgoal_obs"]
        subgoal_probs = L.get_output(self.policy.l_subgoal_prob, {self.policy.l_obs: subgoal_obs})
        bottleneck_probs = L.get_output(self.policy.l_bottleneck_prob, {self.policy.l_obs: obs_var})
        N = tf.shape(obs_var)[0]
        p_z = tf.reduce_mean(bottleneck_probs, reduction_indices=0)
        p_g_z = tf.matmul(tf.transpose(subgoal_probs), bottleneck_probs) / tf.cast(N, tf.float32)
        p_g_given_z = p_g_z / tf.expand_dims(p_z, 0)
        # p_g_given_z, p_g_z, p_z = self.p_g_given_z_sym(obs_var, state_info_vars)

        # retrieve p(a|g,z), which is a tensor of dimension |Z|*|G|*|A|
        # we first transpose it so it's |G|*|Z|*|A|
        p_a_given_g_z = tf.transpose(self.policy.prob_tensor, (1, 0, 2))

        # Now, compute the entropy so we get H(A|g,z). This should be a matrix of size |G|*|Z|
        ent_A_given_g_z = tf.reshape(
            self.policy.distribution.entropy_sym(dict(prob=tf.reshape(p_a_given_g_z, (-1, action_dim)))),
            (self.policy.subgoal_dim, self.policy.bottleneck_dim)
        )

        # Now, we take expectation to obtain H(A|G,Z). This should be a single scalar
        ent_A_given_G_Z = tf.reduce_sum(p_g_z * ent_A_given_g_z)

        # compute p(a|z). we just need to marginalize p(a|g,z) over p(g|z). should be a matrix of size |Z|*|A|
        p_a_given_z = tf.reduce_sum(p_a_given_g_z * tf.expand_dims(p_g_given_z, 2), 0)
        # compute H(A|z)
        ent_A_given_z = self.policy.distribution.entropy_sym(dict(prob=p_a_given_z))
        # compute H(A|Z)
        ent_A_given_Z = tf.reduce_sum(ent_A_given_z * p_z)

        # a bunch of diagnostic terms
        p_g = tf.reduce_sum(p_g_z, reduction_indices=1)
        p_z_given_g = p_g_z / tf.expand_dims(p_g, 1)
        p_a_given_g = tf.reduce_sum(p_a_given_g_z * tf.expand_dims(p_z_given_g, 2), 1)
        ent_A_given_g = self.policy.distribution.entropy_sym(dict(prob=p_a_given_g))
        ent_A_given_G = tf.reduce_sum(ent_A_given_g * p_g)

        ent_G_given_z = self.policy.subgoal_dist.entropy_sym(dict(prob=tf.transpose(p_g_given_z)))
        ent_G_given_Z = tf.reduce_sum(ent_G_given_z * p_z)
        ent_Z = tf.reduce_mean(self.policy.bottleneck_dist.entropy_sym(dict(prob=tf.reshape(p_z, (1, -1)))))
        ent_G = tf.reduce_mean(self.policy.subgoal_dist.entropy_sym(dict(prob=tf.reshape(p_g, (1, -1)))))

        ent_Z_given_g = self.policy.bottleneck_dist.entropy_sym(dict(prob=p_z_given_g))
        ent_Z_given_G = tf.reduce_sum(ent_Z_given_g * p_g)
        ent_G_given_S = tf.reduce_mean(self.policy.subgoal_dist.entropy_sym(dict(prob=subgoal_probs)))
        ent_Z_given_S = tf.reduce_mean(self.policy.bottleneck_dist.entropy_sym(dict(prob=bottleneck_probs)))

        mi = ent_A_given_Z - ent_A_given_G_Z

        self.log_items = [
            ("I(A;G|Z)", mi),
            ("H(A|Z)", ent_A_given_Z),
            ("H(A|G,Z)", ent_A_given_G_Z),
            ("H(A|G)", ent_A_given_G),
            ("I(A;Z|G)", ent_A_given_G - ent_A_given_G_Z),
            ("H(G|Z)", ent_G_given_Z),
            ("H(Z)", ent_Z),
            ("H(G)", ent_G),
            ("H(Z|G)", ent_Z_given_G),
            ("H(G|S)", ent_G_given_S),
            ("H(Z|S)", ent_Z_given_S),
        ]

        self.f_logs = tensor_utils.compile_function(
            inputs=[obs_var, state_info_vars["subgoal_obs"]],
            outputs=[x[1] for x in self.log_items]
        )

        return mi

    def bonus_sym(self, obs_var, state_info_vars):
        return self.mi_a_g_given_z_sym(obs_var, state_info_vars)

    def log_diagnostics(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        subgoal_obs = np.concatenate([p["agent_infos"]['subgoal_obs'] for p in paths])
        log_vals = self.f_logs(observations, subgoal_obs)
        for (k, _), v in zip(self.log_items, log_vals):
            logger.record_tabular(k, v)


class HierPolopt(BatchPolopt):
    def __init__(self,
                 env,
                 policy,
                 aux_policy,
                 bonus_evaluator=None,
                 mi_coeff=0.0,
                 step_size=0.01,
                 imitation_max_epochs=10,
                 **kwargs):
        self.mi_coeff = mi_coeff
        self.aux_policy = aux_policy
        if bonus_evaluator is None:
            bonus_evaluator = BonusEvaluator(env_spec=env.spec, policy=policy)
        self.aux_imitation_trainer = PolicyImitationTrainer(max_epochs=imitation_max_epochs)
        self.aux_improvement_trainer = PolicyImprovementTrainer()
        self.hier_imitation_trainer = BonusImitationTrainer(bonus_evaluator, mi_coeff, max_epochs=imitation_max_epochs)
        self.bonus_evaluator = bonus_evaluator
        self.step_size = step_size
        BatchPolopt.__init__(self, env=env, policy=policy, **kwargs)

    @overrides
    def init_opt(self):
        assert not self.policy.recurrent
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        self.aux_imitation_trainer.update_opt(self.aux_policy, obs_var, state_info_vars_list, old_dist_info_vars_list)
        self.aux_improvement_trainer.update_opt(self.aux_policy, self.step_size, obs_var, action_var, advantage_var,
                                                state_info_vars_list, old_dist_info_vars_list)
        self.hier_imitation_trainer.update_opt(self.policy, obs_var, state_info_vars_list, old_dist_info_vars_list)

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        hier_kl = tf.reduce_mean(self.policy.distribution.kl_sym(old_dist_info_vars, dist_info_vars))
        self.f_hier_kl = tensor_utils.compile_function(
            inputs=[obs_var] + state_info_vars_list + old_dist_info_vars_list,
            outputs=hier_kl
        )

        return dict()

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    @overrides
    def optimize_policy(self, itr, samples_data):
        observations = samples_data["observations"]
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        # 1. train the flat policy to match the action distribution on the collected samples
        with logger.prefix("AuxImitation | "), logger.tabular_prefix("AuxImitation"):
            self.aux_imitation_trainer.train([samples_data["observations"]] + state_info_list + dist_info_list)

        # compute dist_info on the training data by the flat policy
        # TODO: should we do this?
        aux_dist_info = self.aux_policy.dist_info(observations, agent_infos)
        aux_dist_info_list = [aux_dist_info[k] for k in self.aux_policy.distribution.dist_info_keys]

        # 2. run policy improvement on the flat policy, as if the samples were collected using the flat policy instead
        with logger.prefix("AuxImprovement | "), logger.tabular_prefix("AuxImprovement"):
            self.aux_improvement_trainer.train(list(all_input_values) + state_info_list + aux_dist_info_list)

        # 3. train the hierarchical policy to match the improved action distribution (no bonus for now,
        # and see whether this can preserve performance)
        new_aux_dist_info = self.aux_policy.dist_info(observations, agent_infos)
        new_aux_dist_info_list = [new_aux_dist_info[k] for k in self.aux_policy.distribution.dist_info_keys]

        with logger.prefix("HierImitation | "), logger.tabular_prefix("HierImitation"):
            self.hier_imitation_trainer.train([samples_data["observations"]] + state_info_list + new_aux_dist_info_list)

        # compute the final KL
        final_kl = self.f_hier_kl(*([samples_data["observations"]] + state_info_list + dist_info_list))
        logger.record_tabular("FinalMeanKL", final_kl)

        self.bonus_evaluator.log_diagnostics(samples_data["paths"])

        return dict()
