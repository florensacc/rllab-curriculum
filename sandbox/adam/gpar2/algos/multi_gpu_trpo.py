# from rllab.algos.npo import NPO
# from sandbox.adam.gpar.algos.npo import NPO
from sandbox.adam.gpar2.algos.multi_gpu_npo import MultiGpuNPO
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.adam.gpar2.optimizers.parallel_conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer
# from rllab.core.serializable import Serializable
# from rllab.misc import ext
from rllab.misc.overrides import overrides
import multiprocessing as mp
import numpy as np
from sandbox.adam.util import struct

# import psutil

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

    def initialize_par_objs(self):
        """ Assumes float32 for all floatX data """
        n = self.n_gpu
        d = self.batch_size // n  # floor division
        self.data_per_worker = d
        obs_size = int(self.env.observation_space.flat_dim)
        act_size = int(self.env.action_space.flat_dim)
        grad_size, agent_info_shapes = self.get_policy_dims()

        n_arr = np.ctypeslib.as_array  # shortcut
        m_arr = mp.RawArray
        obs = [n_arr(m_arr('f', d * obs_size)).reshape(d, obs_size)
                for _ in range(n - 1)]
        act = [n_arr(m_arr('f', d * act_size)).reshape(d, act_size)
                for _ in range(n - 1)]
        adv = [n_arr(m_arr('f', d)) for _ in range(n - 1)]
        agent_infos = list()
        for rank in range(n - 1):
            agent_infos_rank = struct()
            # assume float32 is good for all these
            for k, v_shape in agent_info_shapes.items():
                if not v_shape:  # i.e. scalar, v_shape = ()
                    v_size = 1
                else:
                    v_size = int(np.prod(v_shape))
                agent_infos_rank[k] = n_arr(m_arr('f', d * v_size)).reshape(d, *v_shape)
            agent_infos.append(agent_infos_rank)

        par_objs_master = struct(
            obs=obs,
            act=act,
            adv=adv,
            agent_infos=agent_infos,
        )
        par_objs_ranks = []
        for rank in range(n - 1):
            par_obj_rank = struct(
                obs=obs[rank],
                act=act[rank],
                adv=adv[rank],
                agent_infos=agent_infos[rank],
            )
            par_objs_ranks.append(par_obj_rank)

        #  All optimizer par_objs are shared by all threads, so let them inherit.
        self.optimizer.initialize_par_objs(
            n_parallel=self.n_gpu,
            grad_size=grad_size,
        )

        return par_objs_master, par_objs_ranks

    def get_policy_dims(self):
        """ Spawn a throw-away process in which to instantiate the policy """
        grad_size = mp.RawValue('i')
        manager = mp.Manager()
        agent_info_shapes = manager.dict()  # don't get type for now, assume 'f'
        p = mp.Process(target=self.policy_dim_getter,
            args=(grad_size, agent_info_shapes))
        p.start()
        p.join()
        grad_size = int(grad_size.value)  # strip multiprocessing types
        agent_info_shapes = agent_info_shapes.copy()
        return grad_size, agent_info_shapes

    def policy_dim_getter(self, grad_size, agent_info_shapes):
        """ To run on a sub-process (instantiates the policy) """
        self.instantiate_policy()
        grad_size.value = len(self.policy.get_param_values(trainable=True))
        o = self.env.reset()
        a, agent_info = self.policy.get_action(o)
        for k, v in agent_info.items():
            v_np = np.asarray(v)
            if v_np.dtype == 'object':
                raise TypeError("Unsupported agent_info data structure under key: {}"
                    "\n  (must be able to cast under np.asarray() and not result in "
                    "dtype=='object')".format(k))
            agent_info_shapes[k] = v_np.shape

    def prepare_opt_inputs(self, my_data):
        all_input_values = tuple([my_data.obs, my_data.act, my_data.adv])
        print('copy test:\n')
        print('are they the same object: ', all_input_values[0] is my_data.obs)
        agent_infos = my_data.agent_infos
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            raise NotImplementedError
            # all_input_values += (samples_data["valids"],)  # not supported
        return all_input_values

    def assign_data(self, samples_data):
        """ called by master only """
        obs = samples_data["observations"]
        act = samples_data["actions"]
        adv = samples_data["advantages"]
        agent_infos = samples_data["agent_infos"]
        d = self.data_per_worker
        c = 0
        for w in range(self.n_gpu - 1):
            self.par_objs.obs[w][:] = obs[c:c + d]
            self.par_objs.act[w][:] = act[c:c + d]
            self.par_objs.adv[w][:] = adv[c:c + d]
            for k, v in agent_infos.items():
                self.par_objs.agent_infos[w][k][:] = v[c:c + d]
            c += d

        # Currently, any extra data at the end is left out, for load balancing:
        my_agent_infos = dict()
        for k, v in agent_infos.items():
            my_agent_infos[k] = v[c:c + d]
        my_data = struct(
            obs=obs[c:c + d],  # (works because d was made with floor division)
            act=act[c:c + d],
            adv=adv[c:c + d],
            agent_infos=my_agent_infos,
        )

        return my_data

    @gt.wrap
    @overrides
    def optimize_policy(self, itr, samples_data):
        my_data = self.assign_data(samples_data)
        gt.stamp('assign_data')
        self.par_objs.loop_ctrl.barrier.wait()  # signal workers to re-enter
        my_input_values = self.prepare_opt_inputs(my_data)
        gt.stamp('prep_inputs')
        self.optimizer.optimize(my_input_values)
        gt.stamp('optimize')
        return dict()

    @overrides
    def optimize_policy_worker(self, rank):
        # master writes to assigned_paths (worker barrier outside this method)
        my_input_values = self.prepare_opt_inputs(self.par_objs)
        self.optimizer.optimize_worker(my_input_values)
