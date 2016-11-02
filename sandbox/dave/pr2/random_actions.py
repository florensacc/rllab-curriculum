from __future__ import absolute_import
from scipy.stats import uniform

from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2FixedGoalGenerator
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoFixedBlockGenerator
from six.moves import range


train_goal_generator = PR2FixedGoalGenerator()

env = normalize(Pr2EnvLego(goal_generator=train_goal_generator,
                        lego_generator=PR2LegoFixedBlockGenerator(),
                        qvel_init_std=0,
                        pos_normal_sample=False,
                        # pos_normal_sample_std=1,
                        allow_random_restarts=True,
                        allow_random_vel_restarts=True,
                        distance_thresh=0.01,  # 1 cm
                        use_depth=True,
                        use_vision=True,
                       ))

for _ in range(1000):
    env.reset()
    for _ in range(100):
        env.render()
        max_action = 0
        actions = uniform.rvs(-max_action, 2 * max_action, env.action_space.flat_dim)
        # print "Actions: " + str(actions)
        env.step(actions)  # take a random action
        # env.step(env.action_space.sample()) # take a random action
