


from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

from rllab.algos.base import RLAlgorithm
import rllab.plotter as plotter
from rllab.misc import special
from rllab.algos import util
from sandbox.rocky.tf.policies.base import Policy
# from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.misc import tensor_utils

import tensorflow as tf
import numpy as np


class Sampler(VectorizedSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths):
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.algo.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        padded_paths = []
        subgoal_interval = self.algo.policy.subgoal_interval
        for path in paths:
            path_length = len(path["rewards"])
            padded_length = int(np.ceil(1.0 * path_length / subgoal_interval) * subgoal_interval)
            new_p = dict(path, valids=np.ones((path_length,), dtype=np.int))
            if padded_length != path_length:
                new_p = tensor_utils.pad_tensor_dict(new_p, padded_length)
            padded_paths.append(new_p)

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in padded_paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in padded_paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in padded_paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in padded_paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in padded_paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in padded_paths])
        valids = tensor_utils.concat_tensor_list([path["valids"] for path in padded_paths])

        if self.algo.center_adv:
            advantages = util.center_advantages(advantages)

        if self.algo.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.mean(self.algo.policy.distribution.entropy(observations, valids, agent_infos))

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
            valids=valids,
            paths=paths,
            padded_paths=padded_paths,
        )

        logger.log("fitting baseline...")
        self.algo.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        if sampler_cls is None:
            if self.policy.vectorized:
                sampler_cls = Sampler
            else:
                raise NotImplementedError
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.start_worker()
            for itr in range(self.start_itr, self.n_itr):
                with logger.prefix('itr #%d | ' % itr):
                    paths = self.obtain_samples(itr)
                    samples_data = self.process_samples(itr, paths)
                    self.log_diagnostics(paths)
                    self.optimize_policy(itr, samples_data)
                    logger.log("saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("saved")
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                      "continue...")
        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        super(NPO, self).__init__(**kwargs)

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
        valid_var = tensor_utils.new_tensor(
            'valid',
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

        pick_advantage_var = tf.reshape(advantage_var, (-1, self.policy.subgoal_interval))
        # just take the first ones
        pick_advantage_var = pick_advantage_var[:, 0]

        pick_valid_var = tf.reduce_prod(tf.reshape(valid_var, (-1, self.policy.subgoal_interval)), -1)

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)


        mean_kl = tf.reduce_sum(kl * pick_valid_var) / tf.reduce_sum(pick_valid_var)
        surr_loss = - tf.reduce_sum(lr * pick_advantage_var * pick_valid_var) / tf.reduce_sum(pick_valid_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                         valid_var
                     ] + state_info_vars_list + old_dist_info_vars_list

        self.f_debug = tensor_utils.compile_function(
            input_list,
            [kl, lr, pick_advantage_var, pick_valid_var]
        )

        self.optimizer.update_opt(
            loss=surr_loss,
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
            "observations", "actions", "advantages", "valids"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]

        if hasattr(self.policy, "dist_info"):
            dist_info = self.policy.dist_info(samples_data["observations"], agent_infos)
            dist_info_list = [dist_info[k] for k in self.policy.distribution.dist_info_keys]
        else:
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        # data = self.f_debug(*all_input_values)
        # import ipdb; ipdb.set_trace()
        loss_before = self.optimizer.loss(all_input_values)
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)
