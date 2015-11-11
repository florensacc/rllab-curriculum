from path import Path
import sys
import cgtcompat
import cPickle as pickle
import marshal
sys.setrecursionlimit(50000)

if cgtcompat.is_theano():
    import theano
else:
    import cgt

def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z

def extract(x, *keys):
    return tuple(x[k] for k in keys)

def cached_function(inputs, outputs):
    if cgtcompat.is_theano():
        if hasattr(outputs, '__len__'):
            hash_content = tuple(map(theano.pp, outputs))
        else:
            hash_content = theano.pp(outputs)
        cache_key = hex(hash(hash_content) & (2**64-1))[:-1]
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
        fun = theano.function(inputs, outputs, allow_input_downcast=True, on_unused_input='ignore')
        with open(cache_file, "wb") as f:
            pickle.dump(fun, f, protocol=pickle.HIGHEST_PROTOCOL)
        return fun
    else:
        return cgtcompat.function(inputs, outputs)
