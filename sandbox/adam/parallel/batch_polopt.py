
import multiprocessing as mp
import numpy as np

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.misc import ext
from sandbox.adam.parallel.sampler import WorkerBatchSampler
from sandbox.adam.parallel.util import SimpleContainer
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
            whole_paths=True,
            n_parallel=1,
            set_cpu_affinity=False,
            cpu_assignments=None,
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
        self.set_cpu_affinity = set_cpu_affinity
        self.cpu_assignments = cpu_assignments
        self.worker_batch_size = batch_size // n_parallel
        self.sampler = WorkerBatchSampler(self)

    def __getstate__(self):
        #  (multiprocessing does not allow pickling of manager objects)
        return {k: v for k, v in iter(self.__dict__.items()) if k != "_par_objs"}

    #
    # Serial methods.
    # (Either for calling before forking subprocesses, or subprocesses execute
    # it independently of each other.)
    #

    def _init_par_objs_batchpolopt(self):
        """
        Any init_par_objs() method in a derived class must call this method,
        and, following that, may append() the SimpleContainer objects as needed.
        """
        n = self.n_parallel
        par_data = SimpleContainer(rank=None, avg_fac=1.0 / n)
        shareds = SimpleContainer(
            sum_discounted_return=mp.RawArray('d', n),
            sum_return=mp.RawArray('d', n),
            num_traj=mp.RawArray('i', n),
            max_return=mp.RawArray('d', n),
            min_return=mp.RawArray('d', n),
            num_steps=mp.RawArray('i', n),
            num_valids=mp.RawArray('d', n),
            sum_ent=mp.RawArray('d', n),
            n_steps_collected=mp.RawArray('i', n),
        )
        mgr_objs = SimpleContainer(
            barrier_dgnstc=mp.Barrier(self.n_parallel),
            barrier_avgfac=mp.Barrier(self.n_parallel),
        )
        self._par_objs = (par_data, shareds, mgr_objs)
        self.baseline.init_par_objs(n_parallel=self.n_parallel)

    def init_par_objs(self):
        """
        Initialize all objects use for parallelism (called before forking).
        """
        raise NotImplementedError

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

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    #
    # Main external method and its target for parallel subprocesses.
    #

    def train(self):
        self.init_opt()
        self.init_par_objs()
        processes = [mp.Process(target=self._train, args=(rank,))
            for rank in range(self.n_parallel)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def _train(self, rank):
        self.set_rank(rank)
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths, n_steps_collected = self.sampler.obtain_samples(itr)
                self.set_avg_fac(n_steps_collected)  # (parallel)
                samples_data, dgnstc_data = self.sampler.process_samples(itr, paths)
                self.log_diagnostics(itr, samples_data, dgnstc_data)  # (parallel)
                self.optimize_policy(itr, samples_data)  # (parallel)
                if rank == 0:
                    logger.log("fitting baseline...")
                self.baseline.fit(paths)  # (parallel)
                if rank == 0:
                    logger.log("fitted")
                    logger.log("saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)
                    params["algo"] = self
                    if self.store_paths:
                        # NOTE: Only paths from rank==0 worker will be saved.
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("saved")
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                      "continue...")
                self.current_itr = itr + 1

    #
    # Parallelized methods.
    #

    def log_diagnostics(self, itr, samples_data, dgnstc_data):
            par_data, shareds, mgr_objs = self._par_objs

            i = par_data.rank
            shareds.sum_discounted_return[i] = \
                np.sum([path["returns"][0] for path in samples_data["paths"]])
            undiscounted_returns = [sum(path["rewards"]) for path in samples_data["paths"]]
            shareds.num_traj[i] = len(undiscounted_returns)
            shareds.num_steps[i] = sum([len(path["rewards"]) for path in samples_data["paths"]])
            shareds.sum_return[i] = np.sum(undiscounted_returns)
            shareds.min_return[i] = np.min(undiscounted_returns)
            shareds.max_return[i] = np.max(undiscounted_returns)
            if not self.policy.recurrent:
                shareds.sum_ent[i] = np.sum(self.policy.distribution.entropy(
                    samples_data["agent_infos"]))
                shareds.num_valids[i] = 0
            else:
                shareds.sum_ent[i] = np.sum(self.policy.distribution.entropy(
                    samples_data["agent_infos"]) * samples_data["valids"])
                shareds.num_valids[i] = np.sum(samples_data["valids"])

            # TODO: ev needs sharing before computing.
            # ev = special.explained_variance_1d(
            #     np.concatenate(dgnstc_data["baselines"]),
            #     np.concatenate(dgnstc_data["returns"])
            # )

            mgr_objs.barrer_dgnstc.wait()

            if par_data.rank == 0:
                num_traj = sum(shareds.num_traj)
                average_discounted_return = \
                    sum(shareds.sum_discounted_return) / num_traj
                average_return = sum(shareds.sum_return) / num_traj
                if self.policy.recurrent:
                    ent = sum(shareds.sum_ent) / sum(shareds.num_valids)
                else:
                    ent = sum(shareds.sum_ent) / sum(shareds.num_steps)
                max_return = max(shareds.max_return)
                min_return = min(shareds.min_return)

                logger.record_tabular('Iteration', itr)
                logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
                logger.record_tabular('AverageReturn', average_return)
                # logger.record_tabular('ExplainedVariance', ev)
                logger.record_tabular('NumTrajs', num_traj)
                logger.record_tabular('Entropy', ent)
                logger.record_tabular('Perplexity', np.exp(ent))
                # logger.record_tabular('StdReturn', np.std(undiscounted_returns))
                logger.record_tabular('MaxReturn', max_return)
                logger.record_tabular('MinReturn', min_return)

        # NOTE: These others might only work if all path data is collected
        # centrally, could provide this as an option...might be easiest to build
        # multiprocessing pipes to send the data to the rank-0 process, so as
        # not to have to construct shared variables of specific sizes
        # beforehand.
        #
        # self.env.log_diagnostics(paths)
        # self.policy.log_diagnostics(paths)
        # self.baseline.log_diagnostics(paths)

    def set_rank(self, rank):
        par_data, _, _ = self._par_objs
        par_data.rank = rank
        if self.set_cpu_affinity:
            self._set_affinity(rank)
        self.baseline.set_rank(rank)
        self.optimizer.set_rank(rank)
        seed = ext.get_seed()
        if seed is None:
            # NOTE: Not sure if this is a good source for seed?
            seed = int(1e6 * np.random.rand())
        ext.set_seed(seed + rank)

    def set_avg_fac(self, n_steps_collected):
        par_data, shareds, mgr_objs = self._par_objs
        shareds.n_steps_collected[par_data.rank] = n_steps_collected
        mgr_objs.barrier_avgfac.wait()
        avg_fac = 1.0 * n_steps_collected / sum(shareds.n_steps_collected)
        par_data.avg_fac = avg_fac
        self.optimizer.set_avg_fac(avg_fac)

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def _set_affinity(self, rank, verbose=False):
        import psutil
        if self.cpu_assignments is not None:
            n_assignments = len(self.cpu_assignments)
            assigned_affinity = [self.cpu_assignments[rank % n_assignments]]
        else:
            assigned_affinity = [rank % psutil.cpu_count()]
        p = psutil.Process()
        # NOTE: let psutil raise the error if invalid cpu assignment.
        try:
            p.cpu_affinity(assigned_affinity)
            if verbose:
                logger.log("\nRank: {},  CPU Affinity: {}".format(rank, p.cpu_affinity()))
        except AttributeError:
            logger.log("Cannot set CPU affinity (maybe in a Mac OS).")
