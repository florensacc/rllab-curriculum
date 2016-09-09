


import numpy as np

from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import PerlmutterHvp, FiniteDifferenceHvp
from rllab.misc import logger
from rllab.misc import krylov

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc import ext
from rllab.misc.overrides import overrides
import scipy.signal
import tensorflow as tf


class CategoricalLookbackPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            bottleneck_dim=32,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(CategoricalLookbackPolicy, self).__init__(env_spec)

        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs = L.InputLayer(shape=(None, obs_dim), name="obs")
            l_prev_action = L.InputLayer(shape=(None, action_dim), name="action")

            l_bottleneck = L.DenseLayer(
                l_obs,
                num_units=bottleneck_dim,
                nonlinearity=tf.tanh,
                name="bottleneck"
            )
            l_joint = L.concat([l_bottleneck, l_prev_action], axis=1, name="joint")

            l_hidden = L.DenseLayer(
                l_joint,
                num_units=32,
                nonlinearity=tf.tanh,
                name="hidden"
            )

            l_prob = L.DenseLayer(
                l_hidden,
                num_units=action_dim,
                nonlinearity=tf.nn.softmax,
                name="prob"
            )

            self.bottleneck_dim = bottleneck_dim
            self.l_obs = l_obs
            self.l_prev_action = l_prev_action
            self.l_bottleneck = l_bottleneck
            self.l_prob = l_prob

            self.action_dim = action_dim

            self.prev_actions = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            self.f_prob_bottleneck = tensor_utils.compile_function(
                inputs=[self.l_obs.input_var, self.l_prev_action.input_var],
                outputs=L.get_output([self.l_prob, self.l_bottleneck]),
                log_name="f_prob_bottleneck"
            )

            self.regressor = CategoricalMLPRegressor(
                input_shape=(self.bottleneck_dim,),
                output_dim=self.action_dim,
                hidden_nonlinearity=tf.tanh,
                use_trust_region=False,
                name="p_at_given_zt"
            )

            LayersPowered.__init__(self, [l_prob])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        flat_obs_var = tf.reshape(obs_var, (-1, self.observation_space.flat_dim))
        prev_action_var = state_info_vars["prev_action"]
        flat_prev_action_var = tf.reshape(prev_action_var, (-1, self.action_dim))
        flat_prob = L.get_output(self.l_prob, {self.l_obs: flat_obs_var, self.l_prev_action: flat_prev_action_var})
        prob = tf.reshape(flat_prob, tf.pack([tf.shape(obs_var)[0], tf.shape(obs_var)[1], -1]))
        return dict(prob=prob)

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))

        self.prev_actions[dones] = 0.

    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        prev_actions = self.prev_actions
        probs, bottlenecks = self.f_prob_bottleneck(flat_obs, prev_actions)
        actions = special.weighted_sample_n(probs, np.arange(self.action_dim))
        self.prev_actions = self.action_space.flatten_n(actions)
        return actions, dict(prob=probs, prev_action=prev_actions, bottleneck=bottlenecks)

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        return [
            ("prev_action", (self.action_dim,)),
            # ("reward_bonus", (1,))
        ]

    # def reg_sym(self, state_info_vars, action_var, dist_info_vars, old_dist_info_vars, valid_var, obs_var, **kwargs):
    #     lr = self.distribution.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
    #     N = tf.shape(obs_var)[0]
    #     T = tf.shape(obs_var)[1]
    #
    #     action_var = tf.cast(action_var, tf.float32)
    #
    #     prev_actions = state_info_vars["prev_action"]
    #
    #     flat_obs = tf.reshape(obs_var, tf.pack([N * T, self.observation_space.flat_dim]))
    #     flat_prev_actions = tf.reshape(prev_actions, tf.pack([N * T, self.action_dim]))
    #     flat_actions = tf.reshape(action_var, tf.pack([N * T, self.action_dim]))
    #     flat_lr = tf.reshape(lr, (-1,))
    #
    #     flat_bottlenecks, flat_action_probs = L.get_output(
    #         [self.l_bottleneck, self.l_prob],
    #         {self.l_obs: flat_obs, self.l_prev_action: flat_prev_actions}
    #     )
    #
    #     log_p_at_given_zt = self.regressor.log_likelihood_sym(flat_bottlenecks, flat_actions)
    #     #     flat_action_probs * tf.log(L.get_output(self.regressor._l_prob, flat_bottlenecks) + 1e-8), -1
    #     # )
    #     log_p_at_given_zt_aprev = tf.reduce_sum(flat_action_probs * tf.log(flat_action_probs + 1e-8), -1)
    #     flat_valid_var = tf.reshape(valid_var, (-1,))
    #     mi_bonus = log_p_at_given_zt_aprev - log_p_at_given_zt
    #
    #     mean_mi_bonus = tf.reduce_sum(flat_valid_var * mi_bonus) / tf.reduce_sum(flat_valid_var)
    #
    #     surr_mi_bonus = log_p_at_given_zt_aprev - flat_lr * tf.stop_gradient(log_p_at_given_zt) - log_p_at_given_zt
    #
    #     surr_mi_loss = - self.mi_coeff * (tf.reduce_sum(surr_mi_bonus) / tf.reduce_sum(flat_valid_var))
    #
    #     return surr_mi_loss

    def compute_bonus(self, samples_data):
        # fit the regressor for p(at|zt)
        paths = samples_data["paths"]
        valid_agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        valid_actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        valid_bottlenecks = valid_agent_infos["bottleneck"]
        valid_probs = valid_agent_infos["prob"]

        logger.log("fitting p(at|zt) regressor...")

        # only fit the valid part
        self.regressor.fit(valid_bottlenecks, valid_actions)

        all_bottlenecks = samples_data["agent_infos"]["bottleneck"]
        all_actions = samples_data["actions"]
        flat_bottlenecks = all_bottlenecks.reshape((-1, self.bottleneck_dim))
        flat_actions = all_actions.reshape((-1, self.action_dim))
        log_p_at_given_zt = self.regressor.predict_log_likelihood(flat_bottlenecks, flat_actions)
        log_p_at_given_zt = log_p_at_given_zt.reshape((all_bottlenecks.shape[0], all_bottlenecks.shape[1]))
        log_p_at_given_zt_aprev = np.log(np.sum(all_actions * samples_data["agent_infos"]["prob"], axis=-1) + 1e-8)

        mi_bonus = log_p_at_given_zt_aprev - log_p_at_given_zt

        valid_ent_at_given_zt = np.mean(-self.regressor.predict_log_likelihood(valid_bottlenecks, valid_actions))
        valid_ent_at_given_zt_aprev = np.mean(np.sum(- valid_probs * np.log(valid_probs + 1e-8), axis=-1))

        valid_mi = valid_ent_at_given_zt - valid_ent_at_given_zt_aprev
        logger.record_tabular("H(at|zt)", valid_ent_at_given_zt)
        logger.record_tabular("H(at|zt,at-1)", valid_ent_at_given_zt_aprev)
        logger.record_tabular("I(at;at-1|zt)", valid_mi)
        return mi_bonus


class ConjugateGradient(Serializable):
    """
    This class only computes the natural descent direction via conjugate gradient without performing further steps.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            hvp_approach=None,
    ):
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp()
        self._hvp_approach = hvp_approach

    def update_opt(self, loss, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint", *args,
                   **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
        that the first dimension of these inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which should not be subsampled
        :return: No return value.
        """

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        constraint_term, constraint_value = leq_constraint

        params = target.get_params(trainable=True)
        grads = tf.gradients(loss, xs=params)
        for idx, (grad, param) in enumerate(zip(grads, params)):
            if grad is None:
                grads[idx] = tf.zeros_like(param)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        self._hvp_approach.update_opt(f=constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
            ),
            f_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
            ),
            f_loss_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
                log_name="f_loss_constraint",
            ),
        )

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def constraint_val(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_constraint"](*(inputs + extra_inputs))

    def compute_direction(self, inputs, extra_inputs=None, subsample_grouped_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self._subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("computing descent direction")

        flat_g = self._opt_fun["f_grad"](*(inputs + extra_inputs))

        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.

        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        return flat_descent_step, initial_step_size


class BiobjectiveConjugateGradientOptimizer(Serializable):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            loss1_coeff,
            loss2_coeff,
            cg1=None,
            cg2=None,
            backtrack_ratio=0.8,
            max_backtracks=15,
            accept_violation=False):
        """
        :type cg1: ConjugateGradient
        :type cg2: ConjugateGradient
        """
        Serializable.quick_init(self, locals())
        self.loss1_coeff = loss1_coeff
        self.loss2_coeff = loss2_coeff
        self.cg1 = cg1
        self.cg2 = cg2
        self.target = None
        self.max_constraint_val = None
        self.backtrack_ratio = backtrack_ratio
        self.max_backtracks = max_backtracks
        self.accept_violation = accept_violation

    def update_opt(self, loss1, loss2, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint",
                   *args, **kwargs):
        self.target = target
        self.max_constraint_val = leq_constraint[1]
        self.cg1.update_opt(loss=loss1, target=target, leq_constraint=leq_constraint, inputs=inputs,
                            extra_inputs=extra_inputs, constraint_name=constraint_name, *args, **kwargs)
        self.cg2.update_opt(loss=loss2, target=target, leq_constraint=leq_constraint, inputs=inputs,
                            extra_inputs=extra_inputs, constraint_name=constraint_name, *args, **kwargs)

    def loss1(self, inputs, extra_inputs=None):
        return self.cg1.loss(inputs, extra_inputs)

    def loss2(self, inputs, extra_inputs=None):
        return self.cg2.loss(inputs, extra_inputs)

    def constraint_val(self, inputs, extra_inputs=None):
        return self.cg1.constraint_val(inputs, extra_inputs)

    def optimize(self, inputs, extra_inputs=None):
        with logger.prefix("CG1 | "):
            descent_step1, step_size1 = self.cg1.compute_direction(inputs=inputs, extra_inputs=extra_inputs)
        with logger.prefix("CG2 | "):
            descent_step2, step_size2 = self.cg2.compute_direction(inputs=inputs, extra_inputs=extra_inputs)
        initial_step = self.loss1_coeff * descent_step1 + self.loss2_coeff * descent_step2
        # start from this step and perform backtracking
        prev_param = np.copy(self.target.get_param_values(trainable=True))
        n_iter = 0
        loss1_before = self.cg1.loss(inputs, extra_inputs)
        loss2_before = self.cg2.loss(inputs, extra_inputs)
        for n_iter, ratio in enumerate(self.backtrack_ratio ** np.arange(self.max_backtracks)):
            cur_step = ratio * initial_step
            cur_param = prev_param - cur_step
            self.target.set_param_values(cur_param, trainable=True)

            loss1 = self.cg1.loss(inputs, extra_inputs)
            loss2 = self.cg2.loss(inputs, extra_inputs)
            constraint_val = self.cg1.constraint_val(inputs, extra_inputs)

            if loss1 < loss1_before and loss2 < loss2_before and constraint_val <= self.max_constraint_val:
                break
        if (np.isnan(loss1) or np.isnan(loss2) or np.isnan(constraint_val) or loss1 >= loss1_before or
                    loss2 >= loss2_before or constraint_val >= self.max_constraint_val) and not self.accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss1):
                logger.log("Violated because loss1 is NaN")
            if np.isnan(loss2):
                logger.log("Violated because loss2 is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" % self.constraint_name)
            if loss1 >= loss1_before:
                logger.log("Violated because loss1 not improving")
            if loss2 >= loss2_before:
                logger.log("Violated because loss2 not improving")
            if constraint_val >= self.max_constraint_val:
                logger.log("Violated because constraint %s is violated" % self.constraint_name)
            self.target.set_param_values(prev_param, trainable=True)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("optimization finished")


class Algo(BatchPolopt):
    def __init__(
            self,
            bonus_baseline,
            step_size=0.01,
            mi_coeff=0.,
            **kwargs
    ):
        optimizer = BiobjectiveConjugateGradientOptimizer(
            cg1=ConjugateGradient(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
            cg2=ConjugateGradient(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
            loss1_coeff=1. / (1. + mi_coeff),
            loss2_coeff=mi_coeff / (1. + mi_coeff),
        )
        self.bonus_baseline = bonus_baseline
        self.optimizer = optimizer
        self.step_size = step_size
        super(Algo, self).__init__(**kwargs)

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
        bonus_advantage_var = tensor_utils.new_tensor(
            'bonus_advantage',
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

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                         bonus_advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        if is_recurrent:
            bonus_surr_loss = - tf.reduce_sum(lr * bonus_advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            bonus_surr_loss = - tf.reduce_mean(lr * bonus_advantage_var)

        self.optimizer.update_opt(
            loss1=surr_loss,
            loss2=bonus_surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))

        # compute bonus advantages
        assert isinstance(self.policy, CategoricalLookbackPolicy)

        bonus_rewards = self.policy.compute_bonus(samples_data)
        bonus_rewards = bonus_rewards * samples_data["valids"]
        bonus_paths = []
        valid_bonus_baselines = []
        valid_bonus_returns = []
        valid_bonus_rewards = []

        for path_bonus_rewards, path in zip(bonus_rewards, samples_data["paths"]):
            path_bonus_rewards = path_bonus_rewards[:len(path["rewards"])]
            path = dict(path, rewards=path_bonus_rewards)
            path_baselines = np.append(self.bonus_baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            valid_bonus_baselines.append(path_baselines[:-1])
            valid_bonus_returns.append(path["returns"])
            valid_bonus_rewards.append(path["rewards"])
            bonus_paths.append(path)

        logger.log("fitting bonus baseline...")
        self.bonus_baseline.fit(bonus_paths)
        logger.log("fitted")

        max_path_length = all_input_values[0].shape[1]

        if self.center_adv:
            raw_adv = np.concatenate([path["advantages"] for path in bonus_paths])
            adv_mean = np.mean(raw_adv)
            adv_std = np.std(raw_adv) + 1e-8
            adv = [(path["advantages"] - adv_mean) / adv_std for path in bonus_paths]
        else:
            adv = [path["advantages"] for path in bonus_paths]

        adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

        bonus_ev = special.explained_variance_1d(
            np.concatenate(valid_bonus_baselines),
            np.concatenate(valid_bonus_returns)
        )

        logger.record_tabular('BonusExplainedVariance', bonus_ev)
        logger.record_tabular('BonusAverageReward', np.mean(np.concatenate(valid_bonus_rewards)))

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += (adv,) + tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # first stage optimization
        loss1_before = self.optimizer.loss1(all_input_values)
        loss2_before = self.optimizer.loss2(all_input_values)
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss1_after = self.optimizer.loss1(all_input_values)
        loss2_after = self.optimizer.loss2(all_input_values)
        logger.record_tabular('Loss1Before', loss1_before)
        logger.record_tabular('Loss1After', loss1_after)
        logger.record_tabular('Loss2Before', loss2_before)
        logger.record_tabular('Loss2After', loss2_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss1', loss1_before - loss1_after)
        logger.record_tabular('dLoss2', loss2_before - loss2_after)

        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
