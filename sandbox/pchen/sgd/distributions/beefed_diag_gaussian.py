import theano.tensor as TT
import numpy as np
from rllab.distributions.base import Distribution
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.algos.util import sign


class BeefedDiagGaussian(DiagonalGaussian):

    def grad_log_likelihood_sym(self, x_var, dist_info_vars):
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["log_std"]
        logli = self.log_likelihood_sym(x_var, dist_info_vars)
        return dict(
            mean=(
                1./(TT.square(TT.exp(log_stds))) * (x_var.reshape([-1,self.dim]) - means)
            )
            # log_stds=TT.grad(logli, log_stds),
        )

    def kl_limited_tgt_dist(self, x_var, dist_info_vars, kl, adv):
        # XXX not actually counting change in std
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["log_std"]
        old_std = TT.exp(log_stds)
        grads = self.grad_log_likelihood_sym(x_var, dist_info_vars)
        mean_grad = grads["mean"] * sign(adv.reshape([-1,1]))
        step_size = TT.sqrt(kl / TT.sum(TT.square(mean_grad) / (2.*TT.square(old_std)), axis=1) + 1e-10)
        return dict(
            mean=means+step_size.reshape([-1,1])*mean_grad
        )
