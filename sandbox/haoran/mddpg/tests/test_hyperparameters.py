import unittest

import numpy as np

from misc import hyperparameter as hp
from misc.testing_utils import is_binomial_trial_likely


class TestHyperparameters(unittest.TestCase):
    def test_log_float_param(self):
        param = hp.LogFloatParam("variable", 1e-5, 1e-1)
        n = 10000
        num_success = 0
        threshold = 1e-3
        for _ in range(n):
            if param.generate() > threshold:
                num_success += 1
        p = 0.5
        self.assertTrue(is_binomial_trial_likely(n, p, num_success))

    def test_linear_float_param(self):
        param = hp.LinearFloatParam("variable", -10, 10)
        n = 10000
        num_success = 0
        threshold = 0
        for _ in range(n):
            if param.generate() > threshold:
                num_success += 1
        p = 0.5
        self.assertTrue(is_binomial_trial_likely(n, p, num_success))


class TestHyperparameterSweeper(unittest.TestCase):
    def test_sweep_hyperparameters(self):
        sweeper = hp.HyperparameterSweeper([
            hp.LinearFloatParam("v1", -10, 10),
            hp.LogFloatParam("v2", 1e-5, 1e-1),
        ])
        n = 10000
        num_successes = np.zeros((2, 2))
        threshold_v1 = 0
        threshold_v2 = 1e-3

        def update_success(v1=None, v2=None):
            success_v1 = int(v1 > threshold_v1)
            success_v2 = int(v2 > threshold_v2)
            num_successes[success_v1, success_v2] += 1

        sweeper.sweep_hyperparameters(update_success, n)
        p = 0.25
        for i in range(2):
            for j in range(2):
                self.assertTrue(
                    is_binomial_trial_likely(n, p, num_successes[i, j]))


if __name__ == '__main__':
    unittest.main()
