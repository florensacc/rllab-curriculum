import inspect


class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        spec = inspect.getargspec(self.__init__)
        # Exclude the first "self" parameter
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        varargs = locals_[spec.varargs]
        kwargs = locals_[spec.keywords]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["__args"], **d["__kwargs"])
        self.__dict__.update(out.__dict__)
