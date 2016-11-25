from collections import OrderedDict

import numpy as np


def create_stats_ordered_dict(name, data):
    return OrderedDict([
        (name + 'Mean', np.mean(data)),
        (name + 'Median', np.median(data)),
        (name + 'Std', np.std(data)),
        (name + 'Max', np.max(data)),
        (name + 'Min', np.min(data)),
    ])
