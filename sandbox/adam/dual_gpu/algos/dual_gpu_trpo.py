# from rllab.algos.npo import NPO
# from sandbox.adam.gpar.algos.npo import NPO
from sandbox.adam.dual_gpu.algos.multi_gpu_npo import DualGpuNPO
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


class DualGpuTRPO(DualGpuNPO):
    """
    Trust Region Policy Optimization for two GPUs.

    Concurrent baseline fitter.
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
        super(DualGpuTRPO, self).__init__(optimizer=optimizer, **kwargs)

    def initialize_par_objs(self):
        """ Assumes float32 for all floatX data """
        # n = self.n_gpu
        d = self.batch_size // 2  # n  # floor division
        self.data_per_worker = d
        obs_size = int(self.env.observation_space.flat_dim)
        act_size = int(self.env.action_space.flat_dim)
        grad_size, agent_info_shapes, baseline_size = self.get_policy_and_baseline_dims()
        n_arr = np.ctypeslib.as_array  # shortcut
        m_arr = mp.RawArray

        agent_infos = struct()
        # assume float32 is good for all these
        for k, v_shape in agent_info_shapes.items():
            if not v_shape:  # i.e. scalar, v_shape = ()
                v_size = 1
            else:
                v_size = int(np.prod(v_shape))
            agent_infos[k] = n_arr(m_arr('f', d * v_size)).reshape(d, *v_shape)
        opt_inputs = struct(
            obs=n_arr(m_arr('f', d * obs_size)).reshape(d, obs_size),
            act=n_arr(m_arr('f', d * act_size)).reshape(d, act_size),
            adv=n_arr(m_arr('f', d)),
            agent_infos=agent_infos,
        )

        baseline = struct(
            obs=n_arr(m_arr('f', (d + d) * obs_size)).reshape(d + d, obs_size),
            ret=n_arr(m_arr('f', (d + d))),
            params=n_arr(m_arr('f', baseline_size))
        )

        par_objs = struct(
            opt_inputs=opt_inputs,
            baseline=baseline,
        )

        #  All optimizer par_objs are shared by all threads, so let them inherit.
        self.optimizer.initialize_par_objs(
            n_parallel=self.n_gpu,  # 2
            grad_size=grad_size,
        )

        return par_objs

    def get_policy_and_baseline_dims(self):
        """ Spawn a throw-away process in which to instantiate the policy """
        grad_size = mp.RawValue('i')
        baseline_size = mp.RawValue('i')
        manager = mp.Manager()
        agent_info_shapes = manager.dict()  # don't get type for now, assume 'f'
        p = mp.Process(target=self.policy_and_baseline_dim_getter,
            args=(grad_size, agent_info_shapes, baseline_size))
        p.start()
        p.join()
        grad_size = int(grad_size.value)  # strip multiprocessing types
        baseline_size = int(baseline_size.value)
        agent_info_shapes = agent_info_shapes.copy()
        return grad_size, agent_info_shapes, baseline_size

    def policy_and_baseline_dim_getter(self, grad_size, agent_info_shapes, baseline_size):
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
        self.instantiate_baseline()
        baseline_size.value = len(self.baseline.get_param_values(trainable=True))

    def prepare_opt_inputs(self, my_data):
        all_input_values = tuple([my_data.obs, my_data.act, my_data.adv])
        # print('copy test:\n')
        # print('are they the same object: ', all_input_values[0] is my_data.obs)
        agent_infos = my_data.agent_infos
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            raise NotImplementedError
            # all_input_values += (samples_data["valids"],)  # not supported
        return all_input_values

    def share_data(self, samples_data):
        """ called by master only """

        # for optimization
        obs = samples_data["observations"]
        act = samples_data["actions"]
        adv = samples_data["advantages"]
        agent_infos = samples_data["agent_infos"]
        d = self.data_per_worker

        self.par_objs.opt_inputs.obs[:] = obs[:d]
        self.par_objs.opt_inputs.act[:] = act[:d]
        self.par_objs.opt_inputs.adv[:] = adv[:d]
        for k, v in agent_infos.items():
            self.par_objs.opt_inputs.agent_infos[k][:] = v[:d]

        # Currently, any extra data at the end is left out, for load balancing:
        my_agent_infos = dict()
        for k, v in agent_infos.items():
            my_agent_infos[k] = v[d:d + d]
        my_data = struct(
            obs=obs[d:d + d],  # (works because d was made with floor division)
            act=act[d:d + d],
            adv=adv[d:d + d],
            agent_infos=my_agent_infos,
        )

        # for baseline (all goes to worker)
        self.par_objs.baseline.obs[:] = obs[:d + d]
        self.par_objs.baseline.returns[:] = samples_data["returns"][:d + d]

        return my_data

    @gt.wrap
    @overrides
    def optimize_policy(self, itr, samples_data):
        my_data = self.share_data(samples_data)
        gt.stamp('share_data')
        self.par_objs.loop_ctrl.barrier.wait()  # signal to worker that data was shared
        my_input_values = self.prepare_opt_inputs(my_data)
        gt.stamp('prep_inputs')
        self.optimizer.optimize(my_input_values)
        gt.stamp('optimize')
        return dict()

    @overrides
    def optimize_policy_worker(self):
        # master writes to assigned_paths (worker barrier outside this method)
        my_input_values = self.prepare_opt_inputs(self.par_objs.opt_inputs)
        self.optimizer.optimize_worker(my_input_values)

    @overrides
    def baseline_fit_worker(self):
        """ the point is, this happens while master is sampling """
        baseline_samples_data = dict(
            observations=self.par_objs.baseline.obs,
            returns=self.par_objs.ret,
        )
        self.baseline.fit_by_samples_data(baseline_samples_data)
        self.par_objs.baseline.params[:] = self.baseline.get_param_values(trainable=True)
        # (master will get these params and do the prediction)

