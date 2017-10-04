import inspect
import pickle


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        spec = inspect.getfullargspec(self.__init__)
        # Exclude the first "self" parameter
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        if spec.kwonlyargs:
            kwargs = {kwonlyargs: locals_[kwonlyargs] for kwonlyargs in spec.kwonlyargs}
        else:
            kwargs = dict()
        if spec.varkw is not None and spec.varkw in locals_:
            kwargs = dict(kwargs, **locals_[spec.varkw])
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        in_order_args = inspect.getfullargspec(self.__init__).args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = pickle.loads(pickle.dumps(obj.__getstate__()))
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out
