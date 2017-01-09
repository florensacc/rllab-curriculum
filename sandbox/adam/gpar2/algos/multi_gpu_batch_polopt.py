from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.misc import special
from rllab.misc import tensor_utils
# from rllab.policies.base import Policy
import time
import multiprocessing as mp
import numpy as np
from ctypes import c_bool
from sandbox.adam.util import struct

import gtimer as gt


class BatchSampler(BaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated


class MultiGpuBatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy_cls,
            policy_args,
            baseline,
            network_cls=None,
            network_args=None,
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
            n_gpu=1,
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
        self.policy_cls = policy_cls
        self.policy_args = policy_args
        self.baseline = baseline
        self.network_cls = network_cls
        self.network_args = network_args
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
        self.n_gpu = n_gpu
        if sampler_cls is None:
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)

    def __getstate__(self):
        """ Do not pickle parallel objects """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def initialize_par_objs(self):
        """
        Return any shared objects to pass to workers (avoiding inheritance in
        order to limit which workers have each shared variable on an as-needed
        basis).
        Must be a tuple: first is par_objs for master, second is list of
        par_objs for each rank.
        """
        return (None, [None] * (self.n_gpu - 1))

    def use_gpu(self, gpu_num):
        gpu_str = "gpu" + str(gpu_num)
        import theano.sandbox.cuda  # cannot import theano.sandbox before fork!
        theano.sandbox.cuda.use(gpu_str)

    def initialize_worker(self, rank):
        """ Any set up for an individual optimization worker once it is spawned """
        self.rank = rank
        self.use_gpu(rank)
        self.instantiate_policy()
        # self.optimizer.initialize_rank(rank)  # MUST MATCH PARAM VALUES IN HERE.

    def instantiate_policy(self):
        """
        Called after forking to allow use of multiple GPUs.
        (Overwrite this if using a more complicated policy constructor.)
        """
        if self.network_cls is not None:
            network = self.network_cls(**self.network_args)
            self.policy_args['prob_network'] = network
        self.policy = self.policy_cls(**self.policy_args)

    def start_worker(self):
        sampler_par_objs_master = self.sampler.start_worker()
        par_objs_master, par_objs_ranks = self.initialize_par_objs()
        loop_ctrl = struct(
            shutdown=mp.RawValue(c_bool, False),
            barrier=mp.Barrier(self.n_gpu),
        )
        workers = [mp.Process(target=self.optimizing_worker,
            args=(rank, loop_ctrl, par_objs_ranks[rank]))
            for rank in range(self.n_gpu - 1)]
        for w in workers:
            w.start()
        par_objs_master.loop_ctrl = loop_ctrl  # (don't use these two fields)
        par_objs_master.workers = workers
        self.par_objs = par_objs_master
        self.sampler.par_objs = sampler_par_objs_master  # attach only after fork
        self.initialize_worker(self.n_gpu - 1)  # (master is also a worker)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()
        self.par_objs.loop_ctrl.shutdown.value = True
        self.par_objs.loop_ctrl.barrier.wait()
        for w in self.par_objs.workers:
            w.join()

    def optimizing_worker(self, rank, loop_ctrl, par_objs_rank):
        self.par_objs = par_objs_rank  # make par_objs accessible elsewhere in worker
        self.initialize_worker(rank)
        self.init_opt(rank)
        # self.optimizer.initialize_rank(rank)
        while True:
            loop_ctrl.barrier.wait()
            if not loop_ctrl.shutdown.value:
                self.optimize_policy_worker(rank)
            else:
                break

    def train(self):
        gt.reset_root()
        self.start_worker()
        self.init_opt(rank=self.n_gpu - 1)
        # self.optimizer.initialize_rank(self.n_gpu - 1)
        gt.stamp('init')
        loop = gt.timed_loop('main')
        for itr in range(self.current_itr, self.n_itr):
            next(loop)
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples(itr)
                gt.stamp('paths')
                samples_data = self.sampler.process_samples(itr, paths)
                gt.stamp('samples')
                self.log_diagnostics(paths)
                # self.optimize_policy(itr, samples_data)
                self.fit_baseline(paths, samples_data)
                gt.stamp('fit_baseline')
                self.optimize_policy(itr, samples_data)
                gt.stamp('optimize')
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
        self.shutdown_worker()
        gt.stop()
        print(gt.report())

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

    def optimize_policy_worker(self, rank, **kwargs):
        """ Worker processes execute this method synchrnously with master """
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def fit_baseline(self, paths, samples_data):
        logger.log("fitting baseline...")
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)
        logger.log("fitted")
