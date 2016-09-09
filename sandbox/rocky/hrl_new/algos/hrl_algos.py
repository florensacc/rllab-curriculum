


from sandbox.rocky.tf.algos.npo import NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.envs.base import EnvSpec
import sandbox.rocky.tf.core.layers as L
from rllab.misc import logger

import tensorflow as tf
import numpy as np


class BonusEvaluator(object):
    def __init__(self, env_spec, policy):
        """
        :type env_spec: EnvSpec
        :type policy: FixedClockPolicy
        :return:
        """
        self.env_spec = env_spec
        self.policy = policy

    def p_g_given_z_sym(self, obs_var, state_info_vars):
        subgoal_obs = state_info_vars["subgoal_obs"]
        subgoal_probs = L.get_output(self.policy.l_subgoal_prob, {self.policy.l_obs: subgoal_obs})
        bottleneck_probs = L.get_output(self.policy.l_bottleneck_prob, {self.policy.l_obs: obs_var})
        N = tf.shape(obs_var)[0]
        p_z = tf.reduce_mean(bottleneck_probs, reduction_indices=0)
        p_g_z = tf.matmul(tf.transpose(subgoal_probs), bottleneck_probs) / tf.cast(N, tf.float32)
        p_g_given_z = p_g_z / tf.expand_dims(p_z, 0)
        return p_g_given_z, p_g_z, p_z

    def mi_a_g_given_z_sym(self, obs_var, state_info_vars):
        action_dim = self.policy.action_dim

        # compute p(g|z), p(g,z), and p(z), which are, respectively:
        # p(g|z): a matrix of dimension |G|*|Z|
        # p(g,z): a matrix of dimension |G|*|Z|
        # p(z): a vector of dimension |Z|
        p_g_given_z, p_g_z, p_z = self.p_g_given_z_sym(obs_var, state_info_vars)

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
        ]

        self.f_logs = tensor_utils.compile_function(
            inputs=[obs_var, state_info_vars["subgoal_obs"]],
            outputs=[x[1] for x in self.log_items]
        )

        return mi

    def log_diagnostics(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        subgoal_obs = np.concatenate([p["agent_infos"]['subgoal_obs'] for p in paths])
        log_vals = self.f_logs(observations, subgoal_obs)
        for (k, _), v in zip(self.log_items, log_vals):
            logger.record_tabular(k, v)


class HierNPO(NPO):
    def __init__(self, mi_coeff=1.0, **kwargs):
        self.mi_coeff = mi_coeff
        NPO.__init__(self, **kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

        self.bonus_evaluator = BonusEvaluator(self.env.spec, self.policy)
        mi = self.bonus_evaluator.mi_a_g_given_z_sym(obs_var, state_info_vars)
        surr_loss -= self.mi_coeff * mi

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def log_diagnostics(self, paths):
        NPO.log_diagnostics(self, paths)
        self.bonus_evaluator.log_diagnostics(paths)

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )


class HierTRPO(HierNPO):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 *args, **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        HierNPO.__init__(self, optimizer=optimizer, *args, **kwargs)
