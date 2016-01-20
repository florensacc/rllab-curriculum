from contextlib import contextmanager
from itertools import izip

import theano
import theano.tensor as TT
import numpy as np
from numpy.linalg import LinAlgError

from rllab.algo.natural_gradient_method import NaturalGradientMethod
from rllab.misc import logger, autoargs
from rllab.misc.krylov import cg
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, flatten_hessian, new_tensor, new_tensor_like, \
    flatten_tensor_variables, lazydict
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod


class NPG(NaturalGradientMethod, BatchPolopt, FirstOrderMethod):
    """
    Natural Policy Gradient.
    """

    @autoargs.inherit(NaturalGradientMethod.__init__)
    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    def __init__(
            self,
            **kwargs):
        super(NPG, self).__init__(**kwargs)
        BatchPolopt.__init__(self, **kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        info = super(NPG, self).init_opt(mdp, policy, baseline)
        descent_steps = [
            new_tensor_like("%s descent" % p.name, p)
            for p in policy.params
            ]
        updates = self.update_method(descent_steps, policy.params)
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
