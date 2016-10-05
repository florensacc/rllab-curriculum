
import multiprocessing as mp
import numpy as np
import time

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.misc import ext
from sandbox.haoran.parallel_trpo.sampler import WorkerBatchSampler
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
            serial_compile=True,
            clip_reward=True,
            bonus_evaluator=None,
            extra_bonus_evaluator=None,
            bonus_coeff=0,
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
        self.serial_compile = serial_compile
        self.worker_batch_size = batch_size // n_parallel
        self.n_steps_collected = 0  # (set by sampler)
        self.sampler = WorkerBatchSampler(self)
        self.clip_reward = clip_reward
        self.bonus_evaluator = bonus_evaluator
        if extra_bonus_evaluator is not None:
            raise NotImplementedError
        self.bonus_coeff = bonus_coeff

    def __getstate__(self):
        """ Do not pickle parallel objects. """
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
        self.rank = None
        shareds = SimpleContainer(
            sum_discounted_return=mp.RawArray('d', n),
            num_traj=mp.RawArray('i', n),
            sum_return=mp.RawArray('d', n),
            max_return=mp.RawArray('d', n),
            min_return=mp.RawArray('d', n),
            sum_raw_return=mp.RawArray('d', n),
            max_raw_return=mp.RawArray('d', n),
            min_raw_return=mp.RawArray('d', n),
            num_steps=mp.RawArray('i', n),
            num_valids=mp.RawArray('d', n),
            sum_ent=mp.RawArray('d', n),
        )
        ##HT: for explained variance (yeah I know it's clumsy)
        shareds.append(
            baseline_stats=SimpleContainer(
                y_sum_vec=mp.RawArray('d',n),
                y_square_sum_vec=mp.RawArray('d',n),
                y_pred_error_sum_vec=mp.RawArray('d',n),
                y_pred_error_square_sum_vec=mp.RawArray('d',n),
            )
        )
        barriers = SimpleContainer(
            dgnstc=mp.Barrier(n),
        )
        self._par_objs = (shareds, barriers)
        self.baseline.init_par_objs(n_parallel=n)
        if self.bonus_evaluator is not None:
            self.bonus_evaluator.init_par_objs(n_parallel=n)

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

    def prep_samples(self):
        """
        Used to prepare output from sampler.process_samples() for input to
        optimizer.optimize(), and used in force_compile().
        """
        raise NotImplementedError

    def force_compile(self, n_samples=100):
        """
        Serial - compile Theano (e.g. before spawning subprocesses, if desired)
        """
        logger.log("forcing Theano compilations...")
        paths = self.sampler.obtain_samples(n_samples)
        self.process_paths(paths)
        samples_data, _ = self.sampler.process_samples(paths)
        input_values = self.prep_samples(samples_data)
        self.optimizer.force_compile(input_values)
        self.baseline.force_compile()
        logger.log("all compiling complete")

    #
    # Main external method and its target for parallel subprocesses.
    #

    def train(self):
        self.init_opt()
        if self.serial_compile:
            self.force_compile()
        self.init_par_objs()
        processes = [mp.Process(target=self._train, args=(rank,))
            for rank in range(self.n_parallel)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def process_paths(self, paths):
        for path in paths:
            path["raw_rewards"] = np.copy(path["rewards"])
            if self.clip_reward:
                path["rewards"] = np.clip(path["raw_rewards"],-1,1)
            if self.bonus_evaluator is not None:
                path["bonus_rewards"] = self.bonus_coeff * self.bonus_evaluator.predict(path)
                path["rewards"] = path["rewards"] + path["bonus_rewards"]

    def _train(self, rank):
        self.init_rank(rank)
        print("%d starts _train"%(rank))
        if self.rank == 0:
            start_time = time.time()
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples()
                print("%d: before fitting bonus"%(self.rank))
                if self.bonus_evaluator is not None:
                    if rank == 0:
                        logger.log("fitting bonus evaluator")
                    self.bonus_evaluator.fit_before_process_samples(paths)
                print("%d: after fitting bonus"%(self.rank))
                self.process_paths(paths)
                samples_data, dgnstc_data = self.sampler.process_samples(paths)
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

                    logger.record_tabular("ElapsedTime",time.time()-start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                      "continue...")
                self.current_itr = itr + 1

    #
    # Parallelized methods and related.
    #

    def log_diagnostics(self, itr, samples_data, dgnstc_data):
            shareds, barriers = self._par_objs

            i = self.rank
            shareds.sum_discounted_return[i] = \
                np.sum([path["returns"][0] for path in samples_data["paths"]])
            undiscounted_returns = [sum(path["rewards"]) for path in samples_data["paths"]]
            undiscounted_raw_returns = [sum(path["raw_rewards"]) for path in samples_data["paths"]]
            shareds.num_traj[i] = len(undiscounted_returns)
            shareds.num_steps[i] = self.n_steps_collected
            # shareds.num_steps[i] = sum([len(path["rewards"]) for path in samples_data["paths"]])
            shareds.sum_return[i] = np.sum(undiscounted_returns)
            shareds.min_return[i] = np.min(undiscounted_returns)
            shareds.max_return[i] = np.max(undiscounted_returns)
            shareds.sum_raw_return[i] = np.sum(undiscounted_raw_returns)
            shareds.min_raw_return[i] = np.min(undiscounted_raw_returns)
            shareds.max_raw_return[i] = np.max(undiscounted_raw_returns)
            if not self.policy.recurrent:
                shareds.sum_ent[i] = np.sum(self.policy.distribution.entropy(
                    samples_data["agent_infos"]))
                shareds.num_valids[i] = 0
            else:
                shareds.sum_ent[i] = np.sum(self.policy.distribution.entropy(
                    samples_data["agent_infos"]) * samples_data["valids"])
                shareds.num_valids[i] = np.sum(samples_data["valids"])

            # TODO: ev needs sharing before computing.
            y_pred = np.concatenate(dgnstc_data["baselines"])
            y = np.concatenate(dgnstc_data["returns"])
            shareds.baseline_stats.y_sum_vec[i] = np.sum(y)
            shareds.baseline_stats.y_square_sum_vec[i] = np.sum(y**2)
            shareds.baseline_stats.y_pred_error_sum_vec[i] = np.sum(y-y_pred)
            shareds.baseline_stats.y_pred_error_square_sum_vec[i] = np.sum((y-y_pred)**2)

            barriers.dgnstc.wait()

            if self.rank == 0:
                num_traj = sum(shareds.num_traj)
                average_discounted_return = \
                    sum(shareds.sum_discounted_return) / num_traj
                if self.policy.recurrent:
                    ent = sum(shareds.sum_ent) / sum(shareds.num_valids)
                else:
                    ent = sum(shareds.sum_ent) / sum(shareds.num_steps)
                average_return = sum(shareds.sum_return) / num_traj
                max_return = max(shareds.max_return)
                min_return = min(shareds.min_return)

                average_raw_return = sum(shareds.sum_raw_return) / num_traj
                max_raw_return = max(shareds.max_raw_return)
                min_raw_return = min(shareds.min_raw_return)

                # compute explained variance
                n_steps = sum(shareds.num_steps)
                y_mean = sum(shareds.baseline_stats.y_sum_vec) / n_steps
                y_square_mean = sum(shareds.baseline_stats.y_square_sum_vec) / n_steps
                y_pred_error_mean = sum(shareds.baseline_stats.y_pred_error_sum_vec) / n_steps
                y_pred_error_square_mean = sum(shareds.baseline_stats.y_pred_error_square_sum_vec) / n_steps
                y_var = y_square_mean - y_mean**2
                y_pred_error_var = y_pred_error_square_mean - y_pred_error_mean**2
                if np.isclose(y_var,0):
                    ev = 0 # different from special.exaplained_variance_1d
                else:
                    ev = 1 - y_pred_error_var / (y_var + 1e-8)

                logger.record_tabular('Iteration', itr)
                logger.record_tabular('ExplainedVariance', ev)
                logger.record_tabular('NumTrajs', num_traj)
                logger.record_tabular('Entropy', ent)
                logger.record_tabular('Perplexity', np.exp(ent))
                # logger.record_tabular('StdReturn', np.std(undiscounted_returns))
                logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
                logger.record_tabular('ReturnAverage', average_return)
                logger.record_tabular('ReturnMax', max_return)
                logger.record_tabular('ReturnMin', min_return)
                logger.record_tabular('RawReturnAverage', average_raw_return)
                logger.record_tabular('RawReturnMax', max_raw_return)
                logger.record_tabular('RawReturnMin', min_raw_return)


        # NOTE: These others might only work if all path data is collected
        # centrally, could provide this as an option...might be easiest to build
        # multiprocessing pipes to send the data to the rank-0 process, so as
        # not to have to construct shared variables of specific sizes
        # beforehand.
        #
        # self.env.log_diagnostics(paths)
        # self.policy.log_diagnostics(paths)
        # self.baseline.log_diagnostics(paths)

    def init_rank(self, rank):
        self.rank = rank
        if self.set_cpu_affinity:
            self._set_affinity(rank)
        self.baseline.init_rank(rank)
        self.optimizer.init_rank(rank)
        if self.bonus_evaluator is not None:
            self.bonus_evaluator.init_rank(rank)
        seed = ext.get_seed()
        if seed is None:
            # NOTE: Not sure if this is a good source for seed?
            seed = int(1e6 * np.random.rand())
        ext.set_seed(seed + rank)

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def _set_affinity(self, rank, verbose=False):
        """
        Check your logical cpu vs physical core configuration, use
        cpu_assignments list to put one worker per physical core.  Default
        behavior is to use logical cpus 0,1,2,...
        """
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
