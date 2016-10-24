from sandbox.rocky.neural_learner.envs.doom_hex_goal_finding_maze_env import DoomHexGoalFindingMazeEnv


env = DoomHexGoalFindingMazeEnv(randomize_texture=False)
env.start_interactive()

# path = "/tmp/aaa.wad"
#
# env = DoomHex(full_wad_name=path)
#
# env.start_interactive()
