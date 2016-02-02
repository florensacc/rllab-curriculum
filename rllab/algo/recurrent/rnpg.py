from contextlib import contextmanager
from itertools import izip

import theano
import theano.tensor as TT
import numpy as np
from numpy.linalg import LinAlgError

from rllab.algo.recurrent.recurrent_natural_gradient_method import RecurrentNaturalGradientMethod
from rllab.misc import logger, autoargs
from rllab.misc.krylov import cg
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, flatten_hessian, new_tensor, new_tensor_like, \
    flatten_tensor_variables, lazydict
from rllab.algo.recurrent.recurrent_batch_polopt import RecurrentBatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod


class RNPG(RecurrentNaturalGradientMethod, RecurrentBatchPolopt, FirstOrderMethod):
    """
    Recurrent Natural Policy Gradient.
    """

    @autoargs.inherit(RecurrentNaturalGradientMethod.__init__)
    @autoargs.inherit(RecurrentBatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    def __init__(
            self,
            **kwargs):
        super(RNPG, self).__init__(**kwargs)
        RecurrentBatchPolopt.__init__(self, **kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        info = super(RNPG, self).init_opt(mdp, policy, baseline)
        descent_steps = [
            new_tensor_like("%s descent" % p.name, p)
            for p in policy.get_params(trainable=True)
            ]
        updates = self.update_method(descent_steps, policy.get_params(trainable=True))
        info['f_update'] = lambda: compile_function(
            inputs=descent_steps,
            outputs=None,
            updates=updates,
        )

        return info

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        with self.optimization_setup(itr, policy, samples_data, opt_info) \
                as (_, flat_descent_step):
            f_update = opt_info['f_update']
            descent_steps = policy.flat_to_params(flat_descent_step, trainable=True)
            f_update(*descent_steps)

        return opt_info

