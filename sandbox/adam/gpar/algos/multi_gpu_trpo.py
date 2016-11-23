# from rllab.algos.npo import NPO
# from sandbox.adam.gpar.algos.npo import NPO
from sandbox.adam.gpar.algos.multi_gpu_npo import MultiGpuNPO
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.adam.gpar.optimizers.parallel_conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
import psutil

import gtimer as gt


class MultiGpuTRPO(MultiGpuNPO):
    """
    Trust Region Policy Optimization for multiple GPUs.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ParallelConjugateGradientOptimizer(**optimizer_args)
        super(MultiGpuTRPO, self).__init__(optimizer=optimizer, **kwargs)

    def initialize_par_objs(self, size_grad):
        self.par_objs = struct(
            shutdown=mp.RawValue(c_bool, False),
            barrier=mp.Barrier(self.n_gpu),
            avg_fac=mp.RawArray('d', self.n_gpu),
        )
        manager = mp.Manager()
        assigned_paths = [manager.list() for _ in range(self.n_gpu - 1)]
        workers = [mp.Process(
            target=self.optimizing_worker,
            args=(rank, assigned_paths[rank - 1]))
            for rank in range(1, self.n_gpu)]
        self.par_objs.workers = workers  # workers do not inherit
        self.par_objs.assigned_paths = assigned_paths
        self.optimizer.initialize_par_objs(
            n_parallel=self.n_gpu,
            size_grad=size_grad,
        )

    def prepare_opt_inputs(self, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        return all_input_values

    @gt.wrap
    def assign_paths(self, paths):
        cum_lengths = [0]
        for path in paths:
            cum_lengths.append(cum_lengths[-1] + len(path["rewards"]))
        total_length = cum_lengths[-1]
        data_boundaries = [int(total_length / self.n_gpu) * i
            for i in range(1, self.n_gpu + 1)]
        gt.stamp('boundaries')
        assigned_paths = [list()] * self.n_gpu
        n_samples = [None] * self.n_gpu
        prev_cum_length = 0
        i = 0
        for path, cum_length in zip(paths, cum_lengths[1:]):
            assigned_paths[i].append(path)
            if cum_length > data_boundaries[i]:
                n_samples[i] = cum_length - prev_cum_length
                prev_cum_length = cum_length
                i += 1
        n_samples[-1] = total_length - n_samples[-2]
        gt.stamp('separate')

        # (This is when the paths actually get moved.)
        for i, assignment in enumerate(assigned_paths[1:]):
            self.par_objs.assigned_paths[i][:] = assignment

        for i, n_samp in enumerate(n_samples):
            self.par_objs.avg_fac[i] = 1.0 * n_samp / total_length

        gt.stamp('share')

        return assigned_paths[0]

    @gt.wrap
    @overrides
    def optimize_policy(self, itr, paths):
        p = psutil.Process()
        prev_affinity = p.cpu_affinity()
        p.cpu_affinity([0])
        my_assigned_paths = self.assign_paths(paths)
        gt.stamp('assign_paths')
        self.par_objs.barrier.wait()  # signal workers to re-enter
        my_samples_data, _ = self.sampler.organize_paths(my_assigned_paths)
        gt.stamp('org_paths')
        my_input_values = self.prepare_opt_inputs(my_samples_data)
        gt.stamp('prep_inputs')
        self.optimizer.optimize(my_input_values,
            avg_fac=self.par_objs.avg_fac[0])
        gt.stamp('optimize')
        p.cpu_affinity(prev_affinity)
        return dict()

    def optimize_policy_worker(self, rank, assigned_paths):
        # master writes to assigned_paths
        p = psutil.Process()
        p.cpu_affinity([1])
        my_samples_data, _ = self.sampler.organize_paths(assigned_paths)
        my_input_values = self.prepare_opt_inputs(my_samples_data)
        self.optimizer.optimize_worker(my_input_values,
            avg_fac=self.par_objs.avg_fac[rank])
