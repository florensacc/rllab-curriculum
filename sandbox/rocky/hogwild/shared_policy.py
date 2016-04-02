import multiprocessing as mp
import numpy as np
import cPickle as pickle
from rllab.policies.base import Policy
from sandbox.rocky.hogwild.shared_parameterized import SharedParameterized


def new_shared_mem_array(init_val):
    typecode = init_val.dtype.char
    arr = mp.Array(typecode, np.prod(init_val.shape))
    nparr = np.frombuffer(arr.get_obj(), dtype=init_val.dtype)
    print nparr.shape, init_val.shape
    nparr.shape = init_val.shape
    return nparr


class SharedPolicy(Policy, SharedParameterized):
    def __init__(self, policy):
        SharedParameterized.__init__(self, policy)

    # def __getstate__(self):
    #     return SharedParameterized.__getstate__(self)
    #
    # def __setstate__(self, d):
    #     cls = d["policy_class"]
    #     wrapped_policy = cls(*d["policy_state"]["__args"], **d["policy_state"]["__kwargs"])
    #     for param, param_value in zip(wrapped_policy.get_params(), d["param_values"]):
    #         param.set_value(param_value, borrow=True)
    #
    # def get_params(self, **tags):
    #     return self.wrapped_policy.get_params(**tags)
