from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
from sandbox.dave.pr2.action_limiter import FixedActionLimiter
import numpy as np


class PR2EnvLegoCurriculm(Pr2EnvLego):

    def __init__(self,
            goal_generator=None,
            lego_generator=None,
            action_penalty_weight=0, #originally was 0.001 #there is one with 0.0005
            distance_thresh=0.01,  # 1 cm
            model='pr2_legofree.xml',
            max_action=float("inf"),
            allow_random_restarts=True,   #same position: True
            allow_random_vel_restarts=True,
            qvel_init_std=1, #0.01,
            pos_normal_sample=False,
            pos_normal_sample_std=0.01,
            action_limiter=FixedActionLimiter(),
            use_running_average_failure_rate=False,
            failure_rate_gamma=0.9,
            mean_failure_rate_init=1.0,
            offset=np.zeros(3),
            use_vision=False,
            use_depth=False,
            *args, **kwargs):

        super(PR2EnvLegoCurriculm, self).__init__(goal_generator=None,
                                            lego_generator=None,
                                            action_penalty_weight=0, #originally was 0.001 #there is one with 0.0005
                                            distance_thresh=0.01,  # 1 cm
                                            model='pr2_legofree.xml',
                                            max_action=float("inf"),
                                            allow_random_restarts=True,   #same position: True
                                            allow_random_vel_restarts=True,
                                            qvel_init_std=1, #0.01,
                                            pos_normal_sample=False,
                                            pos_normal_sample_std=0.01,
                                            action_limiter=FixedActionLimiter(),
                                            use_running_average_failure_rate=False,
                                            failure_rate_gamma=0.9,
                                            mean_failure_rate_init=1.0,
                                            offset=np.zeros(3),
                                            use_vision=False,
                                            use_depth=False,
                                            *args, **kwargs,)
