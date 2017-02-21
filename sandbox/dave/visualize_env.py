from rllab.envs.normalized_env import normalize

from sandbox.young_clgan.lib.envs.base import FixedGoalGenerator
from sandbox.young_clgan.lib.envs.maze.point_maze_env import PointMazeEnv
env = normalize(PointMazeEnv(goal_generator=FixedGoalGenerator([0.1, 0.1])))

# from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
# env = normalize(PointMazeEnv())

#env = normalize(PointMazeEnv())
#env = normalize(PointMazeEnv(maze_id=4,length=10))
#env = normalize(AntEnv())
#env = normalize(PointEnv())

while True:
    env.render()
    obs = env.wrapped_env.get_current_obs()
