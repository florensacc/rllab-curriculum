import numpy as np
from rllab.algo.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc.special import explained_variance_1d, discount_cumsum
from rllab.algo.util import center_advantages
import rllab.misc.logger as logger
import rllab.plotter as plotter


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This include various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    @autoargs.arg("n_itr", type=int,
                  help="Number of iterations.")
    @autoargs.arg("start_itr", type=int,
                  help="Starting iteration.")
    @autoargs.arg("batch_size", type=int,
                  help="Number of samples per iteration.")
    @autoargs.arg("max_path_length", type=int,
                  help="Maximum length of a single rollout.")
    @autoargs.arg("whole_paths", type=bool,
                  help="Make sure that the samples contain whole "
                       "trajectories, even if the actual batch size is "
                       "slightly larger than the specified batch_size.")
    @autoargs.arg("discount", type=float,
                  help="Discount.")
    @autoargs.arg("gae_lambda", type=float,
                  help="Lambda used for generalized advantage estimation.")
    @autoargs.arg("center_adv", type=bool,
                  help="Whether to rescale the advantages so that they have "
                       "mean 0 and standard deviation 1")
    @autoargs.arg("record_states", type=bool,
                  help="Whether to record states when sampling")
    @autoargs.arg("store_paths", type=bool,
                  help="Whether to save all paths data to the snapshot")
    @autoargs.arg("plot", type=bool,
                  help="Plot evaluation run after each iteration")
    @autoargs.arg("pause_for_plot", type=bool,
                  help="Plot evaluation run after each iteration")
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
            record_states=False,
            store_paths=False,
            **kwargs
    ):
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
        self.record_states = record_states
        self.store_paths = store_paths

    def start_worker(self, mdp, policy, baseline):
        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

    def shutdown_worker(self):
        pass

    def train(self, mdp, policy, baseline, **kwargs):
        opt_info = self.init_opt(mdp, policy, baseline)
        self.start_worker(mdp, policy, baseline)
        for itr in xrange(self.start_itr, self.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy, baseline)
            opt_info = self.update_baseline(itr, baseline, samples_data,
                                            opt_info)
            opt_info = self.optimize_policy(
                itr, policy, samples_data, opt_info)
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(
                itr, mdp, policy, baseline, samples_data, opt_info)
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
                         opt_info):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def update_plot(self, policy):
        if self.plot:
            plotter.update_plot(policy, self.max_path_length)

    # pylint: disable=R0914
    def obtain_samples(self, itr, mdp, policy, baseline):
        cur_params = policy.get_param_values()
        paths = parallel_sampler.request_samples(
            policy_params=cur_params,
            max_samples=self.batch_size,
            max_path_length=self.max_path_length,
            whole_paths=self.whole_paths,
            record_states=self.record_states
        )

        baselines = []
        returns = []

        for path in paths:
            path["returns"] = discount_cumsum(path["rewards"], self.discount)

        # however, these statistics should still be computed for all paths
        for path in paths:
            path_baselines = np.append(baseline.predict(path), 0)
            deltas = path["rewards"] + \
                self.discount*path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = discount_cumsum(
                deltas, self.discount*self.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.vstack([path["observations"] for path in paths])
        states = np.vstack([path["states"] for path in paths])
        pdists = np.vstack([path["pdists"] for path in paths])
        actions = np.vstack([path["actions"] for path in paths])
        advantages = np.concatenate(
            [path["advantages"] for path in paths])

        if self.center_adv:
            advantages = center_advantages(advantages)

        # Compute various quantities for logging
        ent = policy.compute_entropy(pdists)
        ev = explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )
        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])
        returns = [sum(path["rewards"]) for path in paths]

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('AverageReturn',
                              np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('ExplainedVariance', ev)

        mdp.log_extra(logger, paths)
        policy.log_extra(logger, paths)
        baseline.log_extra(logger, paths)

        samples_data = dict(
            observations=observations,
            advantages=advantages,
            actions=actions,
            pdists=pdists,
            paths=paths,
            states=states,
            baseline_params=baseline.get_param_values(),
        )

        return samples_data

    def update_baseline(self, itr, baseline, samples_data, opt_info):
        baseline.fit(samples_data["paths"])
        return opt_info

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        raise NotImplementedError
