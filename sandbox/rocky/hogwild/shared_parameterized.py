import multiprocessing as mp
import numpy as np
import cPickle as pickle
from rllab.core.parameterized import Parameterized


def new_shared_mem_array(init_val):
    typecode = init_val.dtype.char
    arr = mp.Array(typecode, np.prod(init_val.shape))
    nparr = np.frombuffer(arr.get_obj(), dtype=init_val.dtype)
    print nparr.shape, init_val.shape
    nparr.shape = init_val.shape
    return nparr


class SharedParameterized(Parameterized):
    def __init__(self, param_obj):
        clone_obj = pickle.loads(pickle.dumps(param_obj))
        params = clone_obj.get_params()
        for param in params:
            param.set_value(new_shared_mem_array(param.get_value()), borrow=True)
        self.wrapped_obj = clone_obj

    def __getstate__(self):
        obj_state = self.wrapped_obj.__getstate__()
        param_values = [x.get_value(borrow=True) for x in self.wrapped_obj.get_params()]
        return dict(
            obj_state=obj_state,
            obj_class=type(self.wrapped_obj),
            param_values=param_values
        )

    def __setstate__(self, d):
        cls = d["obj_class"]
        obj_state = d["obj_state"]
        wrapped_policy = cls(*obj_state["__args"], **obj_state["__kwargs"])
        for param, param_value in zip(wrapped_policy.get_params(), d["param_values"]):
            param.set_value(param_value, borrow=True)

    def get_params(self, **tags):
        return self.wrapped_obj.get_params(**tags)

    def new_mem_copy(self):
        """
        Create a clone of the current object, with its own shared parameters
        :return:
        """
        clone = pickle.loads(pickle.dumps(self))
        for param in clone.get_params():
            param.set_value(new_shared_mem_array(param.get_value()), borrow=True)
        return clone

