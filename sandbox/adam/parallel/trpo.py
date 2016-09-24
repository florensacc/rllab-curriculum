
import os

import theano
import theano.tensor as TT

from rllab.misc import ext
from rllab.misc.overrides import overrides
# from rllab.algos.batch_polopt import BatchPolopt
# import rllab.misc.logger as logger
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

from sandbox.adam.parallel.batch_polopt import ParallelBatchPolopt
from sandbox.adam.parallel.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer


class ParallelTRPO(ParallelBatchPolopt):
    """
    Parallelized Trust Region Policy Optimization (Synchronous)

    In this class definition, identical to serial case, except:
        - Inherits from parallelized base class
        - Holds a parallelized optimizer
        - Has an init_par_objs() method (working on base class and optimizer)
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            mkl_num_threads=1,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ParallelConjugateGradientOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.mkl_num_threads = mkl_num_threads
        super(ParallelTRPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        """
        Same as normal NPO, except for setting MKL_NUM_THREADS.
        """
        # Set BEFORE Theano compiling; make equal to number of cores per worker.
        os.environ['MKL_NUM_THREADS'] = str(self.mkl_num_threads)

        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [obs_var,
                      action_var,
                      advantage_var,
                      ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def prep_samples(self, samples_data):
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

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = self.prep_samples(samples_data)
        self.optimizer.optimize(all_input_values)  # (all logging moved in here)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    @overrides
    def init_par_objs(self):
        self._init_par_objs_batchpolopt()  # (must do first)
        self.optimizer.init_par_objs(
            n_parallel=self.n_parallel,
            size_grad=len(self.policy.get_param_values(trainable=True)),
        )
