from rllab.algos.base import RLAlgorithm
# from rllab.sampler import parallel_sampler
# from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter

from rllab.misc import ext
import multiprocessing as mp
import numpy as np
import psutil
from sandbox.adam.gpar.sampler.parallel_gpu_sampler import ParallelGpuSampler
from sandbox.adam.gpar.sampler.gpar_multisampler import GParMultiSampler
from sandbox.adam.gpar.sampler.pairwise_gpu_sampler import PairwiseGpuSampler
from sandbox.adam.gpar.sampler.pairwise_gpu_sampler_2 import PairwiseGpuSampler_2
from sandbox.adam.gpar.sampler.pairwise_gpu_sampler_3 import PairwiseGpuSampler_3
from sandbox.adam.gpar.sampler.pairwise_gpu_sampler_4 import PairwiseGpuSampler_4
from rllab.misc import tensor_utils
import cProfile
import gtimer as gt
import time
import copy


class ParallelGpuBatchPolopt(RLAlgorithm):
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
            sampler_cls=None,
            sampler_args=None,
            n_parallel=2,
            n_simulators=1,
            set_cpu_affinity=True,
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
        if sampler_cls is None:
            sampler_cls = ParallelGpuSampler
        if sampler_args is None:
            sampler_args = dict()
        self.n_parallel = n_parallel
        self.n_simulators = n_simulators
        self.set_cpu_affinity = set_cpu_affinity
        self.cpu_assignments = cpu_assignments
        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def train(self):
        # os.environ['THEANO_FLAGS'] = "device=cpu"
        self.train_barrier = mp.Barrier(self.n_parallel + 1)
        par_sim = [mp.Process(target=self.parallel_simulator, args=(rank,))
            for rank in range(self.n_parallel)]
        for sim in par_sim:
            sim.start()
        self.master_train()
        for sim in par_sim:
            sim.join()

    def master_train(self):
        gt.reset_root()
        gt.rename_root('master')
        if self.set_cpu_affinity:
            p = psutil.Process()
            p.cpu_affinity([0])  # Keep us on core / thread 0.
            all_cpus = list(range(psutil.cpu_count()))
        self.init_opt()
        gt.stamp('init_opt')
        self.force_compile()
        gt.stamp('compile')
        loop = gt.timed_loop('train')
        for itr in range(self.current_itr, self.n_itr):
            next(loop)
            with logger.prefix('iter #%d | ' % itr):
                p.cpu_affinity([0])
                self.train_barrier.wait()  # outside obtain_samples: easier profiling
                paths = self.sampler.obtain_samples_master(itr)  # synchronization with simulators in here.
                gt.stamp('obtain_samp')
                p.cpu_affinity(all_cpus)  # for baseline fitting.
                samples_data = self.sampler.process_samples(itr, paths)
                gt.stamp('process_samp')
                # self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                gt.stamp('opt_pol')
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
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
        loop.exit()
        gt.stop()
        print(gt.report())

    def prof_simulator(self, rank):
        if rank == 0:
            cProfile.runctx('self.parallel_simulator()', globals(), locals(), 'sandbox/adam/gpar/profs/sim_pws_individual_semas_2.prof')
        else:
            self.parallel_simulator(rank)

    def parallel_simulator(self, rank=0):
        gt.reset_root()
        gt.rename_root('simulator_' + str(rank))
        self.init_rank(rank)
        gt.stamp('init_rank')
        loop = gt.timed_loop('train')
        for itr in range(self.current_itr, self.n_itr):
            next(loop)
            self.train_barrier.wait()  # outside obtain_samples: easier profiling
            self.sampler.obtain_samples_simulator(rank, itr)  # synchronization with master in here
            gt.stamp('obtain_samp')
        loop.exit()
        gt.stop()
        if rank == 0:
            time.sleep(1)
            print(gt.report())

    def init_rank(self, rank):
        if self.set_cpu_affinity:
            self._set_cpu_affinity(rank, verbose=True)
        seed = ext.get_seed()
        if seed is None:
            # NOTE: not sure if this is a good source for seed?
            seed = int(1e6 * np.random.rand())
        ext.set_seed(seed + rank)
        if isinstance(self.sampler, GParMultiSampler):
            self.envs = [self.env]
            for _ in range(self.n_simulators - 1):
                self.envs.append(copy.deepcopy(self.env))
        elif isinstance(self.sampler, PairwiseGpuSampler):
            self.env_a = self.env
            self.env_b = copy.deepcopy(self.env)
        elif isinstance(self.sampler, PairwiseGpuSampler_2):
            self.env = [self.env]
            self.env.append(copy.deepcopy(self.env[0]))
        elif isinstance(self.sampler, PairwiseGpuSampler_3) or isinstance(self.sampler, PairwiseGpuSampler_4):
            envs_a = [copy.deepcopy(self.env) for _ in range(self.n_simulators)]
            envs_b = [copy.deepcopy(self.env) for _ in range(self.n_simulators)]
            self.envs = (envs_a, envs_b)

    def _set_cpu_affinity(self, rank, verbose=False):
        """
        Check your logical cpu vs physical core configuration, use
        cpu_assignments list to put one worker per physical core.  Default
        behavior is to use logical cpus 0,1,2,...
        """
        # import psutil  # import in file, so subprocesses inherit.
        if self.cpu_assignments is not None:
            n_assignments = len(self.cpu_assignments)
            assigned_affinity = [self.cpu_assignments[rank % n_assignments]]
        else:
            n_cpu = psutil.cpu_count()
            # NOTE: use this scheme if:
            # CPU numbering goes up from 0 to num_cores, one on each core,
            # followed by num_cores + 1 to num_cores * 2, one on the second
            # hyperthread of each core.
            # assigned_affinity = [rank % n_cpu + rank // (n_cpu // 2) + 1]
            r_mod = rank % (n_cpu - 2)
            cpu = r_mod + 1
            if cpu >= (n_cpu // 2):
                cpu += 1
            assigned_affinity = [cpu]
        p = psutil.Process()
        # NOTE: let psutil raise the error if invalid cpu assignment.
        try:
            p.cpu_affinity(assigned_affinity)
            if verbose:
                logger.log("Rank: {},  CPU Affinity: {}".format(rank, p.cpu_affinity()))
        except AttributeError:
            logger.log("Cannot set CPU affinity (maybe in a Mac OS).")

    def force_compile(self):
        logger.log("forcing compilation of all Theano functions")
        logger.log("..compiling policy action getter")
        paths = self.sampler.obtain_example_samples()
        logger.log("..compiling baseline fit (if applicable)")
        samples_data = self.sampler.process_example_samples(paths)
        all_input_values = self.prepare_samples(samples_data)
        logger.log("..compiling optimizer functions")
        self.optimizer.force_compile(all_input_values)
        logger.log("all compilation complete")

    # # OLD
    # def _train(self):
    #     self.start_worker()
    #     self.init_opt()
    #     for itr in range(self.current_itr, self.n_itr):
    #         with logger.prefix('itr #%d | ' % itr):
    #             paths = self.sampler.obtain_samples(itr)
    #             samples_data = self.sampler.process_samples(itr, paths)
    #             self.log_diagnostics(paths)
    #             self.optimize_policy(itr, samples_data)
    #             logger.log("saving snapshot...")
    #             params = self.get_itr_snapshot(itr, samples_data)
    #             self.current_itr = itr + 1
    #             params["algo"] = self
    #             if self.store_paths:
    #                 params["paths"] = samples_data["paths"]
    #             logger.save_itr_params(itr, params)
    #             logger.log("saved")
    #             logger.dump_tabular(with_prefix=False)
    #             if self.plot:
    #                 self.update_plot()
    #                 if self.pause_for_plot:
    #                     input("Plotting evaluation run: Press Enter to "
    #                               "continue...")

    #     self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

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
