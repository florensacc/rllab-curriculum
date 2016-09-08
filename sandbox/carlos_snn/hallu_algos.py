


from rllab.algos.batch_polopt import BatchPolopt
from rllab.algos.npo import NPO
from rllab.algos.trpo import TRPO
from rllab.algos.ppo import PPO
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger

from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.misc import tensor_utils
import theano
import numpy as np
import theano.tensor as TT
from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables

import numpy as np
import time

class BatchPolopt_hallu(BatchPolopt, Serializable):
    def __init__(self, n_samples=0, asym=False, *args, **kwargs):
        """
        :param hallucinator: Hallucinator object used for generating extra samples
        :param self_normalize: whether to normalize the importance weights to sum to one for the same real experience
        :return:
        """
        self.n_samples=n_samples
        self.asym = asym
        self.__dict__ = None # in init_opt we compile a function on this
        Serializable.quick_init(self, locals())
        BatchPolopt.__init__(self, *args, **kwargs)

    def process_samples(self, itr, paths):
        n_original = len(paths)
        # to assess that resampling asymetricly has an effect, we need a different asymetry in 2 runs
        dif = int(time.time())%20
        n_asym = np.random.randint(1+dif,10+dif,size=n_original)
        print (n_asym)
        for i, path in enumerate(paths[:n_original]):
            if self.asym:
                n_samples = self.n_samples + n_asym[i]
            else:
                n_samples = self.n_samples
            for _ in range(n_samples):
                paths.append(path)
        samples_data = BatchPolopt.process_samples(self,itr,paths)
        return samples_data



    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.start_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.obtain_samples(itr)
                samples_data = self.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
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

        self.shutdown_worker()


class NPO_hallu(NPO, BatchPolopt_hallu):
    def __init__(self, *args, **kwargs):
        BatchPolopt_hallu.__init__(self, *args, **kwargs)
        NPO.__init__(self, *args, **kwargs)


    def init_opt(self):
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

        ##
        ll = dist.log_likelihood_sym(action_var, old_dist_info_vars)

        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        # try to save in self a function that returns the lr
        self.__dict__ = lazydict(
            f_lr=lambda: compile_function(input_list, lr, log_name="f_lr"),
            f_ll=lambda: compile_function(input_list, ll, log_name="f_ll")
        )

        # def lr_func(inputs) :
        #     ret = compile_function(input_list, lr, log_name="lr")
        #     return ret(inputs)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    def optimize_policy(self, itr, samples_data):
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
        loss_before = self.optimizer.loss(all_input_values)

        #lr_before = self.func_dict["f_lr"](*all_input_values)
        ll_before = self.__dict__["f_ll"](*all_input_values)
        # print(*ll_before, sep='\n')
        logger.record_tabular('llBefore', ll_before)

        self.optimizer.optimize(all_input_values)
        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()



class TRPO_hallu(NPO_hallu):
    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 *args, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        NPO_hallu.__init__(self, optimizer=optimizer, *args, **kwargs)


####------------------------######
import theano.tensor as TT
import theano
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algos.batch_polopt import BatchPolopt
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.core.serializable import Serializable


class VPG_hallu(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            n_samples=0,
            asym=False,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.n_samples = n_samples
        self.asym = asym
        super(VPG_hallu, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    @overrides
    def init_opt(self):
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

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            max_kl = TT.max(kl * valid_var)
        else:
            surr_obj = - TT.mean(logli * advantage_var)
            mean_kl = TT.mean(kl)
            max_kl = TT.max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(surr_obj, target=self.policy, inputs=input_list)

        f_kl = ext.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    @overrides
    def process_samples(self, itr, paths):
        n_original = len(paths)
        # to assess that resampling asymetricly has an effect, we need a different asymetry in 2 runs
        dif = int(time.time())%20
        n_asym = np.random.randint(1+dif,10+dif,size=n_original)
        for i, path in enumerate(paths[:n_original]):
            if self.asym:
                n_samples = self.n_samples + n_asym[i]
            else:
                n_samples = self.n_samples
            for _ in range(n_samples):
                paths.append(path)
        samples_data = BatchPolopt.process_samples(self,itr,paths)
        return samples_data

    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
