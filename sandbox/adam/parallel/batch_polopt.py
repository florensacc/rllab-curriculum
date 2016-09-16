
from multiprocessing import Process
import psutil

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.misc import ext
from sandbox.adam.parallel.sampler import WorkerBatchSampler
# from rllab.policies.base import Policy


class ParallelBatchPolopt(RLAlgorithm):
    """
    Base class for parallelized batch sampling-based policy optimization methods.
    This includes various parallelized policy gradient methods like vpg, npg, ppo, trpo, etc.

    Here, parallelized is limited to mean: using multiprocessing package.
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
            whole_paths=False,  # Different default from serial
            n_parallel=1,
            cpu_assignments=None,
            seed=1,
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
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.current_itr = start_itr
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
        self.n_parallel = n_parallel
        self.cpu_assignments = cpu_assignments
        self.sampler = WorkerBatchSampler(self, batch_size // n_parallel)
        self.seed = seed

    def _set_affinity(self, rank, verbose=False):
        if self.cpu_assignments is not None:
            n_assignments = len(self.cpu_assignments)
            assigned_affinity = [self.cpu_assignments[rank % n_assignments]]
        else:
            assigned_affinity = [rank % psutil.cpu_count()]
        p = psutil.Process()
        # NOTE: let psutil raise the error if invalid cpu assignment.
        p.cpu_affinity(assigned_affinity)
        if verbose:
            print("\nRank: {},  Affinity: {}".format(rank, p.cpu_affinity()))

    def _init_par_objs(self):
        self.baseline.init_par_objs(n_parallel=self.n_parallel)
        self.optimizer.init_par_objs(
            n_parallel=self.n_parallel,
            size_grad=len(self.policy.get_param_values(trainable=True)),
        )

    def train(self):
        self.init_opt()
        self._init_par_objs()
        processes = [Process(target=self._train, args=(rank,))
            for rank in range(self.n_parallel)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def _train(self, rank):
        self._set_affinity(rank)
        self.baseline.update_rank(rank)
        self.optimizer.update_rank(rank)
        ext.set_seed(self.seed + rank)
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                # TODO: allow unequal steps returned?
                paths, _ = self.sampler.obtain_samples(itr)
                samples_data = self.sampler.process_samples(itr, paths)
                # TODO: self.log_diagnostics(paths)
                self.optimize_policy(rank, itr, samples_data)
                # logger.log("saving snapshot...")
                # params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                # params["algo"] = self
                # if self.store_paths:
                #     params["paths"] = samples_data["paths"]
                # logger.save_itr_params(itr, params)
                # logger.log("saved")
                # logger.dump_tabular(with_prefix=False)
                # if self.plot and rank==0:
                #     self.update_plot()
                #     if self.pause_for_plot:
                #         input("Plotting evaluation run: Press Enter to "
                #                   "continue...")

    # TODO: all diagnostics
    # def log_diagnostics(self, paths):
    #     self.env.log_diagnostics(paths)
    #     self.policy.log_diagnostics(paths)
    #     self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
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
