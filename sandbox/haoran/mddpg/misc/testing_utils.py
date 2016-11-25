import math
import numpy as np


def is_binomial_trial_likely(n, p, num_success):
    mean = n * p
    std = math.sqrt(n * p * (1 - p))
    margin = 2 * std
    return mean - margin < num_success < mean + margin


def are_np_array_lists_equal(np_list1, np_list2, threshold=1e-5):
    return all(are_np_arrays_equal(arr1, arr2, threshold=threshold)
               for arr1, arr2 in zip( np_list1, np_list2))


def are_np_arrays_equal(arr1, arr2, threshold=1e-5):
    if arr1.shape != arr2.shape:
        return False
    return (np.abs(arr1 - arr2) <= threshold).all()
