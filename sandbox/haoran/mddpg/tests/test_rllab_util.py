import unittest

import numpy as np
from misc.rllab_util import split_paths

from misc.testing_utils import are_np_arrays_equal


def create_path(offset):
    return dict(
        rewards=np.array([-1, 0, 1]) + offset,
        actions=np.array([[5], [7], [9]]) + offset,
        observations=np.array([[2], [4], [8]]) + offset,
    )


class TestRllabUtil(unittest.TestCase):
    def assertNpEqual(self, np_arr1, np_arr2):
        self.assertTrue(
            are_np_arrays_equal(np_arr1, np_arr2),
            "Numpy arrays not equal")

    def test_split_paths(self):
        paths = [create_path(0), create_path(1)]
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        expected_rewards = np.array([
            -1, 0, 1, 0, 1, 2
        ])
        expected_terminals = np.array([
            0, 0, 1, 0, 0, 1
        ])
        expected_obs = np.array([
            [2], [4], [8],
            [3], [5], [9],
        ])
        expected_actions = np.array([
            [5], [7], [9],
            [6], [8], [10],
        ])
        expected_next_obs = np.array([
            [4], [8], [0],
            [5], [9], [0],
        ])
        self.assertNpEqual(rewards, expected_rewards)
        self.assertNpEqual(terminals, expected_terminals)
        self.assertNpEqual(obs, expected_obs)
        self.assertNpEqual(actions, expected_actions)
        self.assertNpEqual(next_obs, expected_next_obs)


if __name__ == '__main__':
    unittest.main()
