from __future__ import absolute_import

from path import Path
import sys
import cPickle as pickle
import random

from rllab.misc.console import colorize
from collections import OrderedDict

sys.setrecursionlimit(50000)


def merge_dict(*args):
    if any([isinstance(x, OrderedDict) for x in args]):
        z = OrderedDict()
    else:
        z = dict()
    for x in args:
        z.update(x)
    return z


def extract(x, *keys):
    if isinstance(x, (dict, lazydict)):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def compact(x):
    """
    For a dictionary this removes all None values, and for a list this removes
    all None elements; otherwise it returns the input itself.
    """
    if isinstance(x, dict):
        return dict((k, v) for k, v in x.iteritems() if v is not None)
    elif isinstance(x, list):
        return filter(lambda elem: elem is not None, x)
    return x


def cached_function(inputs, outputs):
    import theano
    if hasattr(outputs, '__len__'):
        hash_content = tuple(map(theano.pp, outputs))
    else:
        hash_content = theano.pp(outputs)
    cache_key = hex(hash(hash_content) & (2 ** 64 - 1))[:-1]
    cache_dir = Path('~/.hierctrl_cache')
    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir_p()
    cache_file = cache_dir / ('%s.pkl' % cache_key)
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                pass
    fun = compile_function(inputs, outputs)
    with open(cache_file, "wb") as f:
        pickle.dump(fun, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fun


# Immutable, lazily evaluated dict
class lazydict(object):
    def __init__(self, **kwargs):
        self._lazy_dict = kwargs
        self._dict = {}

    def __getitem__(self, key):
        if key not in self._dict:
            self._dict[key] = self._lazy_dict[key]()
        return self._dict[key]

    def __setitem__(self, i, y):
        self.set(i, y)

    def get(self, key, default=None):
        if key in self._lazy_dict:
            return self[key]
        return default

    def set(self, key, value):
        self._lazy_dict[key] = value


def iscanl(f, l, base=None):
    started = False
    for x in l:
        if base or started:
            base = f(base, x)
        else:
            base = x
        started = True
        yield base


def iscanr(f, l, base=None):
    started = False
    for x in list(l)[::-1]:
        if base or started:
            base = f(x, base)
        else:
            base = x
        started = True
        yield base


def scanl(f, l, base=None):
    return list(iscanl(f, l, base))


def scanr(f, l, base=None):
    return list(iscanr(f, l, base))


def compile_function(inputs=None, outputs=None, updates=None):
    import theano
    return theano.function(
        inputs=inputs,
        outputs=outputs,
        updates=updates,
        on_unused_input='ignore',
        allow_input_downcast=True
    )


def new_tensor(name, ndim, dtype):
    import theano.tensor as TT
    return TT.TensorType(dtype, (False,) * ndim)(name)

def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.ndim, arr_like.dtype)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def is_iterable(obj):
    return isinstance(obj, basestring) or getattr(obj, '__iter__', False)

# cut the path for any time >= t
def truncate_path(p, t):
    return dict((k, p[k][:t]) for k in p)


def concat_paths(p1, p2):
    import numpy as np
    return dict((k1, np.concatenate([p1[k1], p2[k1]])) for k1 in p1.keys() if k1 in p2)


def path_len(p):
    return len(p["states"])


def shuffled(sequence):
    deck = list(sequence)
    while len(deck):
        i = random.randint(0, len(deck) - 1)  # choose random card
        card = deck[i]  # take the card
        deck[i] = deck[-1]  # put top card in its place
        deck.pop()  # remove top card
        yield card

def set_seed(seed):
    import numpy as np
    import lasagne
    random.seed(seed)
    np.random.seed(seed)
    lasagne.random.set_rng(np.random.RandomState(seed))
    print(
        colorize(
            'using seed %s' % (str(seed)),
            'green'
        )
    )

def flatten_hessian(cost, wrt, consider_constant=None,
            disconnected_inputs='raise', block_diagonal=True):
    """
    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Vector (1-dimensional tensor) 'Variable' or list of
               vectors (1-dimensional tensors) Variables

    :param consider_constant: a list of expressions not to backpropagate
        through

    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    :return: either a instance of Variable or list/tuple of Variables
            (depending upon `wrt`) repressenting the Hessian of the `cost`
            with respect to (elements of) `wrt`. If an element of `wrt` is not
            differentiable with respect to the output, then a zero
            variable is returned. The return value is of same type
            as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    import theano
    from theano.tensor import arange
    # Check inputs have the right format
    import theano.tensor as TT
    from theano import Variable
    from theano import grad
    assert isinstance(cost, Variable), \
        "tensor.hessian expects a Variable as `cost`"
    assert cost.ndim == 0, \
        "tensor.hessian expects a 0 dimensional variable as `cost`"

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    hessians = []
    if not block_diagonal:
        expr = TT.concatenate([
            grad(cost, input, consider_constant=consider_constant,
                    disconnected_inputs=disconnected_inputs).flatten()
            for input in wrt
        ])

    for input in wrt:
        assert isinstance(input, Variable), \
            "tensor.hessian expects a (list of) Variable as `wrt`"
        # assert input.ndim == 1, \
        #     "tensor.hessian expects a (list of) 1 dimensional variable " \
        #     "as `wrt`"
        if block_diagonal:
            expr = grad(cost, input, consider_constant=consider_constant,
                        disconnected_inputs=disconnected_inputs).flatten()

        # It is possible that the inputs are disconnected from expr,
        # even if they are connected to cost.
        # This should not be an error.
        hess, updates = theano.scan(lambda i, y, x: grad(
            y[i],
            x,
            consider_constant=consider_constant,
            disconnected_inputs='ignore').flatten(),
                                    sequences=arange(expr.shape[0]),
                                    non_sequences=[expr, input])
        assert not updates, \
            ("Scan has returned a list of updates. This should not "
             "happen! Report this to theano-users (also include the "
             "script that generated the error)")
        hessians.append(hess)
    if block_diagonal:
        from theano.gradient import format_as
        return format_as(using_list, using_tuple, hessians)
    else:
        return TT.concatenate(hessians, axis=1)

def flatten_tensor_variables(ts):
    import theano.tensor as TT
    return TT.concatenate(map(TT.flatten, ts))

