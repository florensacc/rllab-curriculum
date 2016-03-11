from rllab.algo.base import RLAlgorithm
from rllab.misc.ext import compact
import lasagne.updates
from rllab.misc import autoargs
from functools import partial


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **compact(kwargs))
    else:
        raise NotImplementedError


class FirstOrderMethod(RLAlgorithm):
    """
    Methods that perform online updates without using an optimizer like l-bfgs.
    This is mostly a stub to handle the hyper-parameters passed to the
    underlying online optimization algorithm (e.g. learning rate, rho for
    rmsprop, etc.)
    """

    @autoargs.arg('update_method', type=str, help='Update method.')
    @autoargs.arg('learning_rate', type=float, help='Learning rate.')
    def __init__(
            self,
            update_method='sgd',
            learning_rate=0.01,
            **kwargs):
        self.update_method = parse_update_method(
            update_method,
            learning_rate=learning_rate
        )
