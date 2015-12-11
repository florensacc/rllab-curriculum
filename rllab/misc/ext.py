from path import Path
import sys
import cPickle as pickle

sys.setrecursionlimit(50000)


def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z


def extract(x, *keys):
    return tuple(x[k] for k in keys)


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

    def get(self, key, default=None):
        if key in self._lazy_dict:
            return self[key]
        return default


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


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def is_iterable(obj):
    return isinstance(obj, basestring) or getattr(obj, '__iter__', False)
