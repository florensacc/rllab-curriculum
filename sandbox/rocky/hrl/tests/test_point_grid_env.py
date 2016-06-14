from __future__ import print_function
from __future__ import absolute_import

from nose2.tools import such
from sandbox.rocky.hrl.envs.point_grid_env import safe_move
import numpy as np

with such.A("Point Grid Env") as it:
    @it.should
    def test_safe_move():
        np.testing.assert_allclose(
            safe_move(np.array([0.5, 0.5]), np.array([0.25, 0.25]), block_size=0.25),
            np.array([0.75, 0.75])
        )
        np.testing.assert_allclose(
            safe_move(np.array([0.5, 0.5]), np.array([0.3, 0.3]), block_size=0.25),
            np.array([0.75, 0.75])
        )
        np.testing.assert_allclose(
            safe_move(np.array([0.875, 0.625]), np.array([0, 0.25]), block_size=0.25),
            np.array([0.875, 0.75])
        )
        sqrt2 = np.sqrt(2)
        np.testing.assert_allclose(
            safe_move(np.array([0.9375, 0.6875]), np.array([-sqrt2 * 0.25, sqrt2 * 0.25]), block_size=0.25),
            np.array([0.875, 0.75])
        )

it.createTests(globals())
