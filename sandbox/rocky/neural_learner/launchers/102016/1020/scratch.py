from rllab.sampler.utils import rollout
from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
from sandbox.rocky.neural_learner.envs.doom_hex_goal_finding_maze_env import DoomHexGoalFindingMazeEnv
from sandbox.rocky.neural_learner.envs.doom_hex_utils import plot_textmap
from sandbox.rocky.neural_learner.envs.doom_two_goal_env import DoomTwoGoalEnv, create_maps
from sandbox.rocky.tf.policies.uniform_control_policy import UniformControlPolicy


env = DoomTwoGoalEnv(
    rescale_obs=(30, 40),
    reset_map=True,
)

# map = create_maps()[1]
# plot_textmap(map, 1000, 1000)

# env = DoomGoalFindingMazeEnv(
#     rescale_obs=(30, 40),
#     reset_map=True,
#     # randomize_texture=False,
#     # difficulty=1,
#     # n_repeats=1,
#     # n_trajs=10
# )

env.start_interactive()
# while True:
#     rollout(env, UniformControlPolicy(env.spec), max_path_length=100, animated=True)
