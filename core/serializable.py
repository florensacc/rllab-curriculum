class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def __getstate__(self):
        return {"__args" : self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["__args"], **d["__kwargs"])
        self.__dict__.update(out.__dict__)
