from collections import OrderedDict

import numpy as np


def create_stats_ordered_dict(name, data):
    return OrderedDict([
        (name + 'Mean', np.mean(data)),
        (name + 'Std', np.mean(data)),
        (name + 'Max', np.max(data)),
        (name + 'Min', np.min(data)),
    ])