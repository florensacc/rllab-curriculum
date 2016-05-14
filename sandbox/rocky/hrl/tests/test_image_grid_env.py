from __future__ import print_function
from __future__ import absolute_import

from nose2.tools import such
from sandbox.rocky.hrl.envs.image_grid_env import ImageGridEnv

with such.A("Image Grid Env") as it:
    @it.should
    def test_env():
        env = ImageGridEnv(size=10, subgoal_interval=2)
        env.reset()
        assert env.agent_pos == (0, 0)
        assert env.goal_pos != (0, 0)
        assert env.goal_pos[0] % 2 == 0
        assert env.goal_pos[1] % 2 == 0

        env.goal_pos = (0, 2)
        env.step(env.action_from_direction('right'))
        assert env.agent_pos == (0, 1)
        _, reward, done, _ = env.step(env.action_from_direction('right'))
        assert env.agent_pos == (0, 2)
        assert reward == 1
        assert done

it.createTests(globals())
