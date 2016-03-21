import numpy as np
from rllab.algo.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc import special #import explained_variance_1d, discount_cumsum
from rllab.misc import tensor_utils #import explained_variance_1d, discount_cumsum
from rllab.algo import util# import center_advantages, shift_advantages_to_positive
import rllab.misc.logger as logger
import rllab.plotter as plotter


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            whole_paths=True,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            **kwargs
    ):
        """
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param whole_paths: Make sure that the samples contain whole trajectories, even if the actual batch size is
        slightly larger than the specified batch_size.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.whole_paths = whole_paths
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths

    def start_worker(self, mdp, policy, baseline):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    def shutdown_worker(self):
        pass

    def train(self, mdp, policy, baseline, **kwargs):
        self.start_worker(mdp, policy, baseline)
        opt_info = self.init_opt(mdp, policy, baseline)
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            paths = self.obtain_samples(itr, mdp, policy, **kwargs)
            samples_data = self.process_samples(itr, paths, mdp, policy, baseline, **kwargs)
            opt_info = self.optimize_policy(
                itr, policy, samples_data, opt_info, **kwargs)
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(
                itr, mdp, policy, baseline, samples_data, opt_info, **kwargs)
            if self.store_paths:
                params["paths"] = samples_data["paths"]
            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot(policy)
                if self.pause_for_plot:
                    raw_input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()

    def init_opt(self, mdp, policy, baseline):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info, **kwargs):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        raise NotImplementedError

    def update_plot(self, policy):
        if self.plot:
            plotter.update_plot(policy, self.max_path_length)

    def obtain_samples(self, itr, mdp, policy, **kwargs):
        cur_params = policy.get_param_values()

        parallel_sampler.request_samples(
            policy_params=cur_params,
            max_samples=self.batch_size,
            max_path_length=self.max_path_length,
            whole_paths=self.whole_paths,
        )

        return parallel_sampler.collect_paths()

    def process_samples(self, itr, paths, mdp, policy, baseline, **kwargs):

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(baseline.predict(path), 0)
            deltas = path["rewards"] + \
                self.discount*path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount*self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.vstack([path["observations"] for path in paths])
        pdists = np.vstack([path["pdists"] for path in paths])
        actions = np.vstack([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        if self.center_adv:
            advantages = util.center_advantages(advantages)

        if self.positive_adv:
            advantages = util.shift_advantages_to_positive(advantages)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = policy.compute_entropy(pdists)

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        logger.log("fitting baseline...")
        baseline.fit(paths)
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

        mdp.log_extra(paths)
        policy.log_extra(paths)
        baseline.log_extra(paths)

        # numerical check
        check_param = policy.get_param_values()
        # if np.any(np.isnan(check_param)):
        #     raise ArithmeticError("NaN in params")
        # elif np.any(np.isinf(check_param)):
        #     raise ArithmeticError("InF in params")

        samples_data = dict(
            observations=observations,
            pdists=pdists,
            actions=actions,
            advantages=advantages,
            paths=paths,
        )

        if policy.is_recurrent:
            return self.recurrent_postprocess_samples(samples_data)
        else:
            return samples_data

    def recurrent_postprocess_samples(self, samples_data):
        paths = samples_data["paths"]

        max_path_length = max([len(path["advantages"]) for path in paths])

        # make all paths the same length (pad extra advantages with 0)
        obs = [path["observations"] for path in paths]
        obs = [tensor_utils.pad_tensor(ob, max_path_length, ob[0]) for ob in obs]

        if self.center_adv:
            raw_adv = np.concatenate([path["advantages"] for path in paths])
            adv_mean = np.mean(raw_adv)
            adv_std = np.std(raw_adv) + 1e-8
            adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
        else:
            adv = [path["advantages"] for path in paths]
        adv = [tensor_utils.pad_tensor(a, max_path_length, 0) for a in adv]

        actions = [path["actions"] for path in paths]
        actions = [tensor_utils.pad_tensor(a, max_path_length, a[0]) for a in actions]
        pdists = [path["pdists"] for path in paths]
        pdists = [tensor_utils.pad_tensor(p, max_path_length, p[0]) for p in pdists]

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = [tensor_utils.pad_tensor(v, max_path_length, 0) for v in valids]

        samples_data["observations"] = np.asarray(obs)
        samples_data["advantages"] = np.asarray(adv)
        samples_data["actions"] = np.asarray(actions)
        samples_data["valids"] = np.asarray(valids)
        samples_data["pdists"] = np.asarray(pdists)

        return samples_data

