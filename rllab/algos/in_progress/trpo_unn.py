import numpy as np

from rllab.algos.batch_polopt import BatchPolopt
from rllab.algos.natural_gradient_method import NaturalGradientMethod
from rllab.algos.npg import TNPG
from rllab.misc import autoargs
from rllab.misc.ext import extract
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from nn_uncertainty import prob_nn as prob_nn
import lasagne
import time


class TRPO(NaturalGradientMethod, BatchPolopt):
    """
    Trust Region Policy Optimization
    """

    @autoargs.inherit(NaturalGradientMethod.__init__)
    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.arg("backtrack_ratio", type=float,
                  help="The exponential backtrack factor")
    @autoargs.arg("max_backtracks", type=int,
                  help="The maximum number of exponential backtracks")
    def __init__(self,
                 backtrack_ratio=0.5,
                 max_backtracks=10,
                 **kwargs):
        super(TRPO, self).__init__(**kwargs)
        BatchPolopt.__init__(self, **kwargs)
        self.backtrack_ratio = backtrack_ratio
        self.max_backtracks = max_backtracks

        ################
        ### UNN init ###
        ################
        batch_size = 5000
        n_batches = 50  # temp

        prob_nn.PLOT_OUTPUT = False
        prob_nn.PLOT_WEIGHTS_INDIVIDUAL = False
        prob_nn.PLOT_WEIGHTS_TOTAL = False
        self.pnn = prob_nn.ProbNN(
            n_in=5,  # dim obs + act
            n_hidden=[50],
            n_out=4,
            n_batches=n_batches,
            layers_type=[1, 1],
            trans_func=lasagne.nonlinearities.rectify,
            out_func=lasagne.nonlinearities.linear,
            batch_size=batch_size,
            n_samples=5,
            type='regression',
            prior_sd=0.05
        )

        # Build symbolic network architecture.
        self.pnn.build_network()
        # Build all symbolic stuff around architecture, e.g., loss, prediction
        # functions, training functions,...
        self.pnn.build_model()

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        with self.optimization_setup(itr, policy, samples_data, opt_info) as (
                inputs, flat_descent_step):

            ####################
            ### UNN training ###
            ####################

            # inputs = (o,a), target = o'
            obs = samples_data['observations']
            act = samples_data['actions']
            obs_nxt = obs[1:]
            obs = obs[:-1]
            act = act[:-1]
            _inputs = np.hstack([obs, act])
            _targets = obs_nxt

            _out = self.pnn.pred_fn(_inputs)

            acc = np.abs(_out - _targets)
            acc = np.mean(acc, axis=1)
            acc = np.mean(acc)

            print('absolute difference %s' % acc)

            # Save old params for every update.
#             self.pnn.save_old_params()

            train_err = 0.
            for epoch in xrange(100):
                # Update model weights based on current minibatch.
                _train_err = self.pnn.train_fn(_inputs, _targets)
                train_err += train_err

            _out = self.pnn.pred_fn(_inputs)

            acc = np.abs(_out - _targets)
            acc = np.mean(acc, axis=1)
            acc = np.mean(acc)

            print('absolute difference %s' % acc)

#             print('kl last update %s' % self.pnn.f_kl_div())
#             print('log_q_w %s' %
#                   (self.pnn.f_q_w() / self.pnn.n_batches / self.pnn.batch_size))
#             print('log_p_w %s' %
#                   (self.pnn.f_p_w() / self.pnn.n_batches / self.pnn.batch_size))

            logger.log("performing backtracking")
            f_trpo_info = opt_info['f_trpo_info']
            prev_loss, prev_mean_kl, prev_max_kl = f_trpo_info(*inputs)
            prev_param = policy.get_param_values(trainable=True)
            for n_iter, ratio in enumerate(self.backtrack_ratio ** np.arange(self.max_backtracks)):
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                policy.set_param_values(cur_param, trainable=True)
                loss, mean_kl, max_kl = f_trpo_info(*inputs)
                if loss < prev_loss and mean_kl <= self.step_size:
                    break
            logger.log("backtracking finished")
            logger.record_tabular('BacktrackItr', n_iter)
            logger.record_tabular('MeanKL', mean_kl)
            logger.record_tabular('MaxKL', max_kl)

        return opt_info
