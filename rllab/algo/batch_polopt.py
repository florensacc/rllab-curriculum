import numpy as np
from rllab.algo.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc.ext import extract
from rllab.misc.special import explained_variance_1d, discount_cumsum
from rllab.algo.util import center_advantages
import rllab.misc.logger as logger
import rllab.plotter as plotter


G = parallel_sampler.G


def worker_inject_baseline(baseline):
    G.baseline = baseline


def worker_retrieve_paths():
    return G.paths


def retrieve_paths():
    return sum(parallel_sampler.run_map(worker_retrieve_paths), [])


def worker_compute_paths_returns(opt):
    for path in G.paths:
        path["returns"] = discount_cumsum(path["rewards"], opt.discount)


def worker_retrieve_samples_data():
    return G.samples_data


def aggregate_samples_data():
    samples_datas = parallel_sampler.run_map(worker_retrieve_samples_data)

    observations, states, pdists, actions, advantages, paths = extract(
        samples_datas,
        "observations", "states", "pdists", "actions", "advantages", "paths"
    )
    return dict(
        observations=np.concatenate(observations),
        states=np.concatenate(states),
        pdists=np.concatenate(pdists),
        actions=np.concatenate(actions),
        advantages=np.concatenate(advantages),
        paths=sum(paths, []),
    )


def worker_process_paths(opt):
    try:
        baselines = []
        returns = []
        for path in G.paths:
            path_baselines = np.append(G.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                opt.discount*path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = discount_cumsum(
                deltas, opt.discount*opt.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.vstack([path["observations"] for path in G.paths])
        states = np.vstack([path["states"] for path in G.paths])
        pdists = np.vstack([path["pdists"] for path in G.paths])
        actions = np.vstack([path["actions"] for path in G.paths])
        advantages = np.concatenate(
            [path["advantages"] for path in G.paths])

        if opt.center_adv:
            advantages = center_advantages(advantages)

        G.samples_data = dict(
            observations=observations,
            states=states,
            pdists=pdists,
            actions=actions,
            advantages=advantages,
            paths=G.paths,
        )

        average_discounted_return = \
            np.mean([path["returns"][0] for path in G.paths])

        undiscounted_returns = [sum(path["rewards"]) for path in G.paths]

        return dict(
            average_discounted_return=average_discounted_return,
            average_return=np.mean(undiscounted_returns),
            std_return=np.std(undiscounted_returns),
            max_return=np.max(undiscounted_returns),
            min_return=np.min(undiscounted_returns),
            num_trajs=len(G.paths),
            ent=G.policy.compute_entropy(pdists),
            ev=explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )
        )
    except Exception as e:
        print e
        import traceback
        traceback.print_exc()
        raise 


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
            algorithm_parallelized=False,
            **kwargs
    ):
        super(RLAlgorithm, self).__init__()
        self.opt.n_itr = n_itr
        self.opt.start_itr = start_itr
        self.opt.batch_size = batch_size
        self.opt.max_path_length = max_path_length
        self.opt.discount = discount
        self.opt.gae_lambda = gae_lambda
        self.opt.plot = plot
        self.opt.pause_for_plot = pause_for_plot
        self.opt.whole_paths = whole_paths
        self.opt.center_adv = center_adv
        self.opt.record_states = record_states
        self.opt.store_paths = store_paths
        self.opt.algorithm_parallelized = algorithm_parallelized

    def start_worker(self, mdp, policy, baseline):
        parallel_sampler.populate_task(mdp, policy)
        parallel_sampler.run_map(worker_inject_baseline, baseline)
        if self.opt.plot:
            plotter.init_plot(mdp, policy)

    def shutdown_worker(self):
        pass

    def train(self, mdp, policy, baseline, **kwargs):
        self.start_worker(mdp, policy, baseline)
        opt_info = self.init_opt(mdp, policy, baseline)
        for itr in xrange(self.opt.start_itr, self.opt.n_itr):
            logger.push_prefix('itr #%d | ' % itr)
            samples_data = self.obtain_samples(itr, mdp, policy, baseline)
            opt_info = self.optimize_policy(
                itr, policy, samples_data, opt_info)
            logger.log("saving snapshot...")
            params = self.get_itr_snapshot(
                itr, mdp, policy, baseline, samples_data, opt_info)
            if self.opt.store_paths:
                params["paths"] = samples_data["paths"]
            logger.save_itr_params(itr, params)
            logger.log("saved")
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.opt.plot:
                self.update_plot(policy)
                if self.opt.pause_for_plot:
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

    def optimize_policy(self, itr, policy, samples_data, opt_info):
        raise NotImplementedError

    def update_plot(self, policy):
        if self.opt.plot:
            plotter.update_plot(policy, self.opt.max_path_length)

    def obtain_samples(self, itr, mdp, policy, baseline):
        cur_params = policy.get_param_values()
        parallel_sampler.request_samples(
            policy_params=cur_params,
            max_samples=self.opt.batch_size,
            max_path_length=self.opt.max_path_length,
            whole_paths=self.opt.whole_paths,
            record_states=self.opt.record_states,
        )

        parallel_sampler.run_map(worker_compute_paths_returns, self.opt)

        if baseline.algorithm_parallelized:
            baseline.fit()
        else:
            if self.opt.algorithm_parallelized:
                print "[Warning] Baseline should be parallelized when using a " \
                      "parallel algorithm for best possible performance"
            paths = retrieve_paths()
            baseline.fit(paths)
        results = parallel_sampler.run_map(worker_process_paths, self.opt)

        average_discounted_returns, average_returns, std_returns, max_returns, \
            min_returns, num_trajses, ents, evs = extract(
                results,
                "average_discounted_return", "average_return", "std_return",
                "max_return", "min_return", "num_trajs", "ent", "ev"
            )

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('Entropy', np.mean(ents))
        logger.record_tabular('Perplexity', np.exp(np.mean(ents)))
        logger.record_tabular('AverageReturn',
                              np.mean(average_returns))
        logger.record_tabular('StdReturn',
                              np.mean(std_returns))
        logger.record_tabular('MaxReturn',
                              np.max(max_returns))
        logger.record_tabular('MinReturn',
                              np.min(min_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              np.mean(average_discounted_returns))
        logger.record_tabular('NumTrajs', np.sum(num_trajses))
        logger.record_tabular('ExplainedVariance', np.mean(evs))

        # the log_extra feature is going to be tricky...
        if not self.opt.algorithm_parallelized:
            paths = retrieve_paths()
            mdp.log_extra(logger, paths)
            policy.log_extra(logger, paths)
            baseline.log_extra(logger, paths)
            return aggregate_samples_data()
        else:
            return dict()
