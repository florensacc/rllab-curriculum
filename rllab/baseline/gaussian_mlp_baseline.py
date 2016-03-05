from pydoc import locate

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.misc import autoargs
from rllab.misc import normal_dist
from rllab.misc.ext import compile_function
from rllab.misc.tensor_utils import flatten_tensors
from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides
from rllab.optim import locate_optimizer
import rllab.misc.logger as logger
import numpy as np
import theano
import theano.tensor as TT
import lasagne.layers as L
import lasagne


class GaussianMLPBaseline(Baseline, LasagnePowered, Serializable):

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg("optimizer", type=str,
                  help="Module path to the optimizer. It must support the "
                       "same interface as scipy.optimize.fmin_l_bfgs_b")
    @autoargs.arg("max_opt_itr", type=int,
                  help="Maximum number of batch optimization iterations.")
    def __init__(
            self,
            mdp,
            hidden_sizes=(32, 32),
            nonlinearity='lasagne.nonlinearities.tanh',
            optimizer='theano_penalty_lbfgs',
            max_opt_itr=20,
            step_size=0.01,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianMLPBaseline, self).__init__(mdp)

        self._optimizer = locate_optimizer(optimizer)(
            max_opt_itr=max_opt_itr,
        )

        mean_network = MLP(
            input_shape=mdp.observation_shape,
            output_dim=1,
            nonlinearity=nonlinearity,
            output_nl=None,
        )

        l_mean = mean_network.l_out

        l_log_std = L.ParamLayer(
            mean_network.l_in,
            num_units=1,
            param=lasagne.init.Constant(0.),
            name="output_log_std",
        )

        LasagnePowered.__init__(self, [l_mean, l_log_std])

        obs_var = mean_network.input_var
        returns_var = TT.vector("returns")
        old_means_var = TT.matrix("old_means")
        old_log_stds_var = TT.matrix("old_log_stds")

        means_var = L.get_output(l_mean)
        log_stds_var = L.get_output(l_log_std)

        mean_kl = TT.mean(normal_dist.kl_sym(
            old_means_var, old_log_stds_var, means_var, log_stds_var))

        loss = - TT.mean(normal_dist.log_likelihood_sym(
            returns_var, l_mean.output_var, l_log_std.output_var))

        self._optimizer.update_opt(
            loss=loss,
            target=self,
            leq_constraint=(mean_kl, step_size),
            inputs=[obs_var, returns_var, old_means_var, old_log_stds_var]
        )

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        old_means, old_log_stds = self._f_pdist(observations)

        inputs = [observations, returns, old_means, old_log_stds]

        loss_before = self._optimizer.loss(*inputs)
        logger.record_tabular('vf_LossBefore', loss_before)
        self._optimizer.optimize(*inputs)
        # loss, _, _ = self._optimizer.evaluate_loss()
        # self._optimizer.optimize(inputs)

        # cur_params = self.get_param_values(trainable=True)

        # def evaluate_cost(penalty):
        #     def evaluate(params):
        #         self.set_param_values(params, trainable=True)
        #         val, = self._f_loss(*input_vals)
        #         return val.astype(np.float64)
        #     return evaluate

        # def evaluate_grad(penalty):
        #     def evaluate(params):
        #         self.set_param_values(params, trainable=True)
        #         grad = self._f_grads(*input_vals)
        #         flattened_grad = flatten_tensors(map(np.asarray, grad))
        #         return flattened_grad.astype(np.float64)
        #     return evaluate

        # loss_before = evaluate_cost(0)(cur_params)
        # logger.record_tabular('vf_LossBefore', loss_before)

        # opt_params, _, _ = self._optimizer(
        #     func=evaluate_cost(0), x0=cur_params,
        #     fprime=evaluate_grad(0),
        #     maxiter=self._max_opt_itr
        # )
        # self.set_param_values(opt_params, trainable=True)

        loss_after = self._optimizer.loss(*inputs)
        logger.record_tabular('vf_LossAfter', loss_after)
        logger.record_tabular('vf_dLoss', loss_before - loss_after)


    @overrides
    def predict(self, path):
        return self._f_value(self._features(path))

    @overrides
    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)
