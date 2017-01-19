import joblib

class Picklable(object):
    def __init__(self):
        self.unpicklable_list = []

    def __getstate__(self):
        return dict(
            (k, v)
            for (k, v) in self.__dict__.items()
            if k not in self.unpicklable_list
        )

    def __setstate__(self,state):
        for k,v in state.items():
            setattr(self,k,v)
