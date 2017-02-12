import gym
import gym.scoreboard.scoring
import openai_benchmark


directory = "data/s3/mujoco-1m-trpo-5"

# gym.scoreboard.scoring.benchmark_score_from_local('Mujoco1M-v0', directory)
gym.upload(directory, benchmark_id="Mujoco1M-v0", algorithm_id='rllab-trpo-test')
