from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2FixedGoalGenerator
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoFixedBlockGenerator
from sandbox.ignasi.envs.action_limited_env import ActionLimitedEnv
from sandbox.ignasi.envs.block_pushing.pr2_env_lego_hand import Pr2EnvLego
import numpy as np

init_hand = np.array([0.6,  0.4,  0.5025])
goal_generator = PR2FixedGoalGenerator(goal = (0.6, 0.1, 0.5025)) # second dimension moves block further away vertically
lego_generator = PR2LegoFixedBlockGenerator(block = (0.6, 0.2, 0.5025, 1, 0, 0, 0)) # want block at 0.6 +/- 0.2, , 0.1 +/- 0.4, 0.5025
env = normalize(Pr2EnvLego(
        goal_generator=goal_generator,
        lego_generator=lego_generator,
        max_action=1,
        pos_normal_sample=True,
        qvel_init_std=0, #0.01,
        # pos_normal_sample_std=.01, # ignored i think?
        fixed_target = init_hand, # sets the initial position of the hand to 0.6 0.3
        # allow_random_restarts=True, #ignored i think?
    ))
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    max_path_length=60,
    plot=True,
)
algo.train()
