def get_time_stamp():
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
    return timestamp

def create_dir_if_needed(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def to_iterable(obj):
    if obj is None:
        return None
    if not hasattr(obj, '__iter__') or type(obj) == str:
        return [obj]
    return obj

def get_softmax(x):
    import numpy as np
    # x - 1D array
    x = x.flatten()
    softmax_x = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
    return softmax_x

def row_concat(a,b):
    # Handles empty array, which np.r_ does not
    import numpy as np
    if 0 in a.shape:
        return np.array(b)
    elif 0 in b.shape:
        return np.array(a)
    else:
        return np.r_[a,b]
