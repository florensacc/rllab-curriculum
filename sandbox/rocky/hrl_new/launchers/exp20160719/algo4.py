from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from rllab.misc import logger
from rllab.misc import special
from rllab.algos import util
import tensorflow as tf


class PredictionBonusEvaluator(object):
    def __init__(self, env_spec, regressor_cls=None):
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        if regressor_cls is None:
            regressor_cls = BernoulliMLPRegressor
        self.regressor = regressor_cls(
            input_shape=(obs_dim + action_dim,),
            output_dim=obs_dim,
            name="state_regressor"
        )
        self.bonus_scale = 1.

    def predict(self, path):
        xs = np.concatenate([path["observations"], path["actions"]], axis=1)[:-1]
        ys = path["observations"][1:]
        return np.append(-self.regressor.predict_log_likelihood(xs, ys) / (self.bonus_scale + 1e-8), 0)

    def fit(self, samples_data):
        paths = samples_data["paths"]
        observations = np.concatenate([p["observations"][:-1] for p in paths])
        actions = np.concatenate([p["actions"][:-1] for p in paths])
        next_observations = np.concatenate([p["observations"][1:] for p in paths])
        xs = np.concatenate([observations, actions], axis=1)
        self.regressor.fit(xs, next_observations)
        all_bonuses = -self.regressor.predict_log_likelihood(xs, next_observations)
        self.bonus_scale = np.median(all_bonuses)

    def log_diagnostics(self, samples_data):
        bonuses = np.concatenate(map(self.predict, samples_data["paths"]))
        logger.record_tabular("AverageBonus", np.mean(bonuses))
        logger.record_tabular("MaxBonus", np.max(bonuses))
        logger.record_tabular("MinBonus", np.min(bonuses))
        logger.record_tabular("StdBonus", np.std(bonuses))


class StateGoalSurpriseBonusEvaluator(object):
    """
    Compute the bonus to be I(g;s'|s) - log(p_old(s'|g,s))
    """

    def __init__(
            self,
            env_spec,
            subgoal_dim,
            regressor_cls=None,
            regressor_args=None,
            mi_coeff=1.,
            surprise_coeff=1.):
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        if regressor_cls is None:
            regressor_cls = BernoulliMLPRegressor
        if regressor_args is None:
            regressor_args = dict()
        self.mi_coeff = mi_coeff
        self.surprise_coeff = surprise_coeff
        self.state_given_prev_regressor = regressor_cls(
            input_shape=(obs_dim,),
            output_dim=obs_dim,
            name="state_given_prev_regressor",
            optimizer=FirstOrderOptimizer(max_epochs=10, batch_size=128, verbose=True),
            **regressor_args
        )
        self.old_state_given_goal_prev_regressor = regressor_cls(
            input_shape=(obs_dim + subgoal_dim,),
            output_dim=obs_dim,
            name="old_state_given_goal_prev_regressor",
            optimizer=FirstOrderOptimizer(max_epochs=10, batch_size=128, verbose=True),
            **regressor_args
        )
        self.state_given_goal_prev_regressor = regressor_cls(
            input_shape=(obs_dim + subgoal_dim,),
            output_dim=obs_dim,
            name="state_given_goal_prev_regressor",
            optimizer=FirstOrderOptimizer(max_epochs=10, batch_size=128, verbose=True),
            **regressor_args
        )

    def predict(self, path):
        obs = path["observations"][:-1]
        obs_goal = np.concatenate([path["observations"], path["agent_infos"]["subgoal"]], axis=1)[:-1]
        next_obs = path["observations"][1:]
        log_p_st_given_st1 = self.state_given_prev_regressor.predict_log_likelihood(obs, next_obs)
        log_p_st_given_gt_st1 = self.state_given_goal_prev_regressor.predict_log_likelihood(obs_goal, next_obs)
        log_p_old_st_given_gt_st1 = self.old_state_given_goal_prev_regressor.predict_log_likelihood(obs_goal, next_obs)

        mi_est = log_p_st_given_gt_st1 - log_p_st_given_st1
        surprise_est = -log_p_old_st_given_gt_st1

        bonus = self.mi_coeff * mi_est + self.surprise_coeff * surprise_est

        return np.append(bonus, 0)

    def fit_before_process_samples(self, paths):
        obs = np.concatenate([p["observations"][:-1] for p in paths])
        goal = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
        next_obs = np.concatenate([p["observations"][1:] for p in paths])
        obs_goal = np.concatenate([obs, goal], axis=1)

        self.old_state_given_goal_prev_regressor.set_param_values(
            self.state_given_goal_prev_regressor.get_param_values()
        )

        self.state_given_prev_regressor.fit(obs, next_obs)
        self.state_given_goal_prev_regressor.fit(obs_goal, next_obs)

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, samples_data):
        bonuses = np.concatenate(map(self.predict, samples_data["paths"]))
        logger.record_tabular("AverageBonus", np.mean(bonuses))
        logger.record_tabular("MaxBonus", np.max(bonuses))
        logger.record_tabular("MinBonus", np.min(bonuses))
        logger.record_tabular("StdBonus", np.std(bonuses))

        paths = samples_data["paths"]
        obs = np.concatenate([p["observations"][:-1] for p in paths])
        goal = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
        next_obs = np.concatenate([p["observations"][1:] for p in paths])
        obs_goal = np.concatenate([obs, goal], axis=1)

        ent_state_given_prev = np.mean(-self.state_given_prev_regressor.predict_log_likelihood(obs, next_obs))
        ent_state_given_goal_prev = np.mean(-self.state_given_goal_prev_regressor.predict_log_likelihood(obs_goal,
                                                                                                         next_obs))
        cross_ent_old_state_given_goal_prev = np.mean(-self.old_state_given_goal_prev_regressor.predict_log_likelihood(
            obs_goal, next_obs))
        mi_est = ent_state_given_prev - ent_state_given_goal_prev
        logger.record_tabular("H(st+1|st)", ent_state_given_prev)
        logger.record_tabular("H(st+1|st,gt)", ent_state_given_goal_prev)
        logger.record_tabular("I(st+1;gt|st)", mi_est)
        logger.record_tabular("KL(p_new||p_old)", cross_ent_old_state_given_goal_prev - ent_state_given_goal_prev)


class FixedClockPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
            name,
            subgoal_dim,
            subgoal_interval,
            hidden_sizes=(32, 32),
    ):

        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Discrete)

        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            self.subgoal_network = MLP(
                input_shape=(obs_dim,),
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=tf.nn.tanh,
                output_dim=subgoal_dim,
                output_nonlinearity=tf.nn.softmax,
                name="subgoal_network"
            )

            l_obs = self.subgoal_network.input_layer
            l_subgoal_in = L.InputLayer(
                shape=(None, subgoal_dim),
                name="subgoal",
            )

            self.action_network = MLP(
                input_shape=(obs_dim + subgoal_dim,),
                input_layer=L.concat([l_obs, l_subgoal_in], axis=1, name="action_input"),
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=tf.nn.tanh,
                output_dim=action_dim,
                output_nonlinearity=tf.nn.softmax,
                name="action_network"
            )

            l_subgoal_prob = self.subgoal_network.output_layer
            l_action_prob = self.action_network.output_layer

            obs_var = l_obs.input_var
            subgoal_in_var = l_subgoal_in.input_var

            subgoal_space = Discrete(subgoal_dim)

            # record current execution time steps
            self.ts = None
            # record current subgoals
            self.subgoals = None
            self.subgoal_probs = None
            self.subgoal_obs = None

            self.subgoal_space = subgoal_space
            self.subgoal_dim = subgoal_dim
            self.subgoal_interval = subgoal_interval

            self.obs_dim = obs_dim
            self.action_dim = action_dim

            self.l_obs = l_obs
            self.l_subgoal_in = l_subgoal_in

            self.l_subgoal_prob = l_subgoal_prob
            self.l_action_prob = l_action_prob

            self.subgoal_dist = Categorical(subgoal_dim)
            self.action_dist = Categorical(action_dim)

            StochasticPolicy.__init__(self, env_spec)

            self.f_subgoal_prob = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=L.get_output(l_subgoal_prob),
            )
            self.f_action_prob = tensor_utils.compile_function(
                inputs=[obs_var, subgoal_in_var],
                outputs=L.get_output(l_action_prob),
            )

            LayersPowered.__init__(self, [self.l_subgoal_prob, l_action_prob])

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        actions, infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in infos.iteritems()}

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.ts is None or len(dones) != len(self.ts):
            self.ts = np.array([-1] * len(dones))
            self.subgoals = np.zeros((len(dones), self.subgoal_dim))
            self.subgoal_probs = np.zeros((len(dones), self.subgoal_dim))
            self.subgoal_obs = np.zeros((len(dones), self.obs_dim))
        self.ts[dones] = -1
        self.subgoals[dones] = np.nan
        self.subgoal_probs[dones] = np.nan
        self.subgoal_obs[dones] = np.nan

    def get_subgoals(self, flat_obs):
        probs = self.f_subgoal_prob(flat_obs)
        subgoals = np.cast['int'](special.weighted_sample_n(probs, np.arange(self.subgoal_dim)))
        return subgoals, probs

    @property
    def distribution(self):
        return self

    @property
    def state_info_specs(self):
        return [
            ("subgoal", (self.subgoal_dim,)),
            ("subgoal_obs", (self.obs_dim,)),
            ("update_mask", (1,)),
        ]

    @property
    def dist_info_keys(self):
        return [k for k, _ in self.dist_info_specs]

    @property
    def dist_info_specs(self):
        return [
            ("subgoal", (self.subgoal_dim,)),
            ("action_prob", (self.action_dim,)),
            ("subgoal_prob", (self.subgoal_dim,)),
            ("update_mask", (1,)),
        ]

    def dist_info_sym(self, obs_var, state_info_vars):
        subgoal_prob = L.get_output(self.l_subgoal_prob, {self.l_obs: state_info_vars["subgoal_obs"]})
        action_prob = L.get_output(self.l_action_prob, {self.l_obs: obs_var, self.l_subgoal_in: state_info_vars[
            "subgoal"]})
        return dict(
            action_prob=action_prob,
            subgoal_prob=subgoal_prob,
            update_mask=state_info_vars["update_mask"],
            subgoal=state_info_vars["subgoal"]
        )

    def entropy(self, dist_info):
        return self.action_dist.entropy(dict(prob=dist_info["action_prob"]))

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_subgoal_prob = old_dist_info_vars["subgoal_prob"]
        old_action_prob = old_dist_info_vars["action_prob"]
        new_subgoal_prob = new_dist_info_vars["subgoal_prob"]
        new_action_prob = new_dist_info_vars["action_prob"]
        update_mask = tf.reshape(old_dist_info_vars["update_mask"], (-1,))
        subgoal_kl = self.subgoal_dist.kl_sym(dict(prob=old_subgoal_prob), dict(prob=new_subgoal_prob))
        action_kl = self.action_dist.kl_sym(dict(prob=old_action_prob), dict(prob=new_action_prob))
        return subgoal_kl * update_mask + action_kl

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        old_subgoal_prob = old_dist_info_vars["subgoal_prob"]
        old_action_prob = old_dist_info_vars["action_prob"]
        new_subgoal_prob = new_dist_info_vars["subgoal_prob"]
        new_action_prob = new_dist_info_vars["action_prob"]
        subgoal_var = old_dist_info_vars["subgoal"]
        update_mask = tf.reshape(old_dist_info_vars["update_mask"], (-1,))
        subgoal_lr = self.subgoal_dist.likelihood_ratio_sym(subgoal_var,
                                                            dict(prob=old_subgoal_prob),
                                                            dict(prob=new_subgoal_prob))
        action_lr = self.action_dist.likelihood_ratio_sym(action_var,
                                                          dict(prob=old_action_prob),
                                                          dict(prob=new_action_prob))
        return (subgoal_lr * update_mask + (1 - update_mask)) * action_lr

    def get_actions(self, observations):
        self.ts += 1
        flat_obs = self.observation_space.flatten_n(observations)
        subgoals, subgoal_probs = self.get_subgoals(flat_obs)
        update_mask = self.ts % self.subgoal_interval == 0
        self.subgoals[update_mask] = self.subgoal_space.flatten_n(subgoals[update_mask])
        self.subgoal_probs[update_mask] = subgoal_probs[update_mask]
        self.subgoal_obs[update_mask] = flat_obs[update_mask]

        # instead of explicitly sampling bottlenecks, we directly marginalize over the distribution to get p(a|g)
        action_probs = self.f_action_prob(flat_obs, self.subgoals)

        actions = special.weighted_sample_n(action_probs, np.arange(self.action_dim))

        return actions, dict(
            action_prob=action_probs,
            subgoal_prob=np.copy(self.subgoal_probs),
            subgoal=np.copy(self.subgoals),
            subgoal_obs=np.copy(self.subgoal_obs),
            update_mask=np.expand_dims(np.cast['int'](update_mask), -1),
        )


class BonusTRPO(TRPO):
    def __init__(self, bonus_evaluator, bonus_coeff, *args, **kwargs):
        self.bonus_evaluator = bonus_evaluator
        self.bonus_coeff = bonus_coeff
        super(BonusTRPO, self).__init__(*args, **kwargs)

    def log_diagnostics(self, samples_data):
        super(BonusTRPO, self).log_diagnostics(samples_data)
        self.bonus_evaluator.log_diagnostics(samples_data)

    def process_samples(self, itr, paths):
        logger.log("fitting bonus evaluator before processing...")
        self.bonus_evaluator.fit_before_process_samples(paths)
        logger.log("fitted")

        # recompute the advantages
        # self.bonus_evaluator.
        baselines = []
        returns = []
        for path in paths:
            bonuses = self.bonus_evaluator.predict(path)
            path["raw_rewards"] = path["rewards"]
            path["rewards"] = path["raw_rewards"] + self.bonus_coeff * bonuses
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            path["raw_returns"] = special.discount_cumsum(path["raw_rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["raw_returns"][0] for path in paths])

            undiscounted_returns = [sum(path["raw_rewards"]) for path in paths]
            undiscounted_bonus_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["raw_rewards"]) for path in paths]
            undiscounted_bonus_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("fitting baseline...")
        self.baseline.fit(paths)
        logger.log("fitted")

        logger.log("fitting bonus evaluator after processing...")
        self.bonus_evaluator.fit_after_process_samples(samples_data)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('AverageBonusReturn', np.mean(undiscounted_bonus_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
