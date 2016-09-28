from sandbox.rocky.tf.envs.base import TfEnv
import contextlib
import random
import numpy as np


def unwrap(env):
    if isinstance(env, TfEnv):
        return unwrap(env.wrapped_env)
    return env


@contextlib.contextmanager
def using_seed(seed):
    rand_state = random.getstate()
    np_rand_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    yield
    random.setstate(rand_state)
    np.random.set_state(np_rand_state)
