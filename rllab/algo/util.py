import numpy as np
import lasagne.updates
from functools import partial
from rllab.misc.ext import compact


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **compact(kwargs))
    else:
        raise NotImplementedError


def center_advantages(advantages):
    return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)
