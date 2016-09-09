

import errno
import os
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
import traceback


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)



@contextmanager
def skip_if_exception():
    try:
        yield
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
