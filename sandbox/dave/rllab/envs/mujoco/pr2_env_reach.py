from __future__ import print_function
from sandbox.dave.rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from scipy.misc import imsave
from sandbox.dave.rllab.mujoco_py.mjviewer_openai import MjViewer
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
from scipy.misc import imresize
import time
# import cv2
import pdb
np.set_printoptions(threshold=np.nan, linewidth=np.nan)
class Pr2EnvLego(MujocoEnv, Serializable):

    FILE = 'pr2_legofree.xml'

    def __init__(
            self,
            goal_generator=None,
            lego_generator=None,
            action_penalty_weight= 0.001, #originally was 0.001 #there is one with 0.0005
            distance_thresh=0.01,  # 1 cm
            model='pr2_legofree.xml', #'pr2_1arm.xml',
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

        self.action_penalty_weight = action_penalty_weight
        self.distance_thresh = distance_thresh
        self.counter = 1
        self._goal_generator = goal_generator
        self._lego_generator = lego_generator
        self._action_limiter = action_limiter
        self.allow_random_restarts = allow_random_restarts
        self.allow_random_vel_restarts = allow_random_vel_restarts
        self.goal_dims = 3
        self.first_time = True
        self.goal = None
        self.lego = None
        if model not in [None, 0]:
            self.set_model(model)
        self.action_limit = max_action
        self.max_action_limit = 3
        self.min_action_limit = 0.1
        self.qvel_init_std = qvel_init_std
        self.pos_normal_sample = pos_normal_sample
        self.pos_normal_sample_std=pos_normal_sample_std
        self.mean_failure_rate = mean_failure_rate_init
        self.failure_rate_gamma = failure_rate_gamma
        self.use_running_average_failure_rate = use_running_average_failure_rate
        self.offset = offset
        self.distance_tip_lego_penalty_weight = 0.6 #0.5 #0.4 #1  #0.1  #0.3
        self.angle_penalty_weight = 0.2 #0.2 #0.4 #0.5 #1 #0.05
        self.occlusion_weight = 0.0005 #0.0005
        self.use_vision = use_vision
        self.use_depth = use_depth
        self.discount = 0.95
        self.depth = np.zeros([99, 99, 3])  #TODO: Hacky
        self.model = model
        self.position_controller = True
        self.roll_joints = [2, 4]
        self.action = np.zeros((7,))
        self.counter = 0
        self.first_action = True
        self.discount_weights = 0.99

        super(Pr2EnvLego, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def set_model(self, model):
        self.__class__.FILE = model

    def get_current_obs(self):
        dim = self.model.data.qpos.shape[0]

        idxpos = list(range(7))  # TODO: Hacky
        idxvel = list(range(7))
        return np.concatenate([
            self.model.data.qpos.flat[idxpos],
            self.goal,                          # We do not need to explicitly include the goal
            #                                   # since we already have the vec to the goal.
            self.model.data.qvel.flat[idxvel],  # Do not include the velocity of the target (should be 0).
            self.get_tip_position(),
            self.get_vec_tip_to_goal(),
        ]).reshape(-1)


    def get_tip_position(self):
        return self.model.data.site_xpos[0]

    def get_lego_position(self):
        return self.model.data.site_xpos[-1]

    def get_vec_tip_to_goal(self):
        tip_position = self.get_tip_position()
        goal_position = self.goal
        vec_tip_to_goal = goal_position - tip_position
        return vec_tip_to_goal

    def step(self, action):
        #action /= 10

        # Limit actions to the specified range.
        if self.use_depth and self.use_vision:
            self.depth = self.viewer.get_depth_map().astype(np.float32)


        vec_tip_to_goal = self.get_vec_tip_to_goal()

        self.forward_dynamics(action)

        distance_tip_to_goal = np.linalg.norm(vec_tip_to_goal)

        # Penalize the robot for being far from the goal and for having the arm far from the lego.
        reward_tip = - distance_tip_to_goal
        # print(reward_tip)

        reward = reward_tip #reward_ctrl#+ reward_occlusion
        state = self._state
        # print(reward_occlusion, reward_angle, reward_tip, reward_dist, )
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self.get_current_obs()


        # Viewer
        if self.use_vision:
            self.viewer.loop_once()

        return Step(ob, float(reward), done, #not self.do_rand,
                    distance_tip_to_goal=distance_tip_to_goal,
                    reward_tip=reward_tip,
                    # reward_occlusion=reward_occlusion,
                    # error_position_x=error_position[0],
                    # error_position_y=error_position[1],
                    # error_position_z=error_position[2],
                    )

    def viewer_setup(self, is_bot=False):
        if self.use_vision:
            self.viewer.cam.camid = -1
            self.viewer.cam.distance = self.model.stat.extent * 1.5
        else:
            self.viewer.cam.camid = -1

    @overrides
    def reset_mujoco(self, qpos=None, qvel=None):
        goal_dims = 3 # self.goal_dims
        lego_dims = 6
        if self.allow_random_restarts or self.first_time:
            if self.pos_normal_sample:
                qpos = self.init_qpos + np.random.normal(size=self.init_qpos.shape) * self.pos_normal_sample_std
            else:
                # Sample a new random initial robot position uniformly from the full joint limit range
                qpos = np.zeros(self.model.data.qpos.shape)
                for idx, jnt_range in enumerate(self.model.jnt_range):
                    qpos[idx] = np.random.uniform(jnt_range[0], jnt_range[1])
                # Make sure joints are within limits
            for idx, jnt_range in enumerate(self.model.jnt_range):
                if self.model.jnt_limited[idx] == 1:
                    qpos[idx] = max(jnt_range[0], qpos[idx])
                    qpos[idx] = min(jnt_range[1], qpos[idx])

        elif qpos is None:
            # Use current position as new position.
            qpos_curr = self.model.data.qpos #[:-goal_dims]
            qpos = list(qpos_curr)
        # Generate a new goal.
        lego_position = self.get_lego_position()

        if self._lego_generator is not None:
            self.lego = self._lego_generator.generate_goal(lego_position)
            qpos[-goal_dims - lego_dims - 1:-goal_dims] = self.lego[:, None]
        else:
        # print("No lego generator!")
            qpos[-goal_dims - lego_dims - 1:-goal_dims] = np.array((0.6, 0.5, 0.5025, 1, 0, 0, 0))[:, None]

        if self._goal_generator is not None:
            self.goal = self._goal_generator.generate_goal(lego_position[:goal_dims])
            qpos[-goal_dims:] = self.goal[:goal_dims, None]
        else:
            print("No goal generator!")

        if self.allow_random_vel_restarts or self.first_time:
            # Generate a new random robot velocity.
            #qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 0.1
            qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * self.qvel_init_std
            #qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * 10
        elif qvel is None:
            qvel = np.array(self.model.data.qvel)

        # Set the velocity of the goal (the goal itself -
        # this is NOT the arm velocity at the goal position!) to 0.
        qvel[-goal_dims-lego_dims:] = 0

        #The position of a free body has 7 components (3 space and 4 for quaternions)

        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

        if self._action_limiter is not None:
            self.action_limit = self._action_limiter.get_action_limit()

        self.first_time = False
        self.first_action = True

        #Apply a force in the Lego block
        xfrc = np.zeros(self.model.data.xfrc_applied.shape)
        weight = 0.1
        xfrc[-2, 2] = -9.81 * weight
        xfrc[13, 2] = - 9.81 * 0.0917
        self.model.data.xfrc_applied = xfrc
        #Viewer

        if self.viewer is None and self.use_vision:
            self.viewer = MjViewer(visible=True, go_fast=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()


        # Choose an action limit from the range of action limits.
        #self.action_limit = np.random.uniform(self.min_action_limit, self.max_action_limit)

    def update_env(self, env_vars):
        self._goal_generator = env_vars.goal_generator
        self._action_limiter = env_vars.action_limiter

    def update_failure_rate(self, paths):
        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        failure_rate = 1 - paths_within_thresh
        if self.use_running_average_failure_rate:
            self.mean_failure_rate = self.mean_failure_rate * self.failure_rate_gamma + failure_rate * (1 - self.failure_rate_gamma)
        else:
            # We don't want to be dependent on the initial failure rate, so just use a large batch size.
            self.mean_failure_rate = failure_rate

    def get_mean_failure_rate(self):
        return self.mean_failure_rate

    @overrides
    def log_diagnostics(self, paths):
        actions = [path["actions"] for path in paths]
        logger.record_tabular('MeanAbsActions', np.mean(np.abs(actions)))
        logger.record_tabular('MaxAbsActions', np.max(np.abs(actions)))
        logger.record_tabular('StdAbsActions', np.std(np.abs(actions)))
        logger.record_tabular('StdActions', np.std(actions))


        distances_tip_to_lego = [path["env_infos"]["distance_tip_to_goal"] for path in paths]
        logger.record_tabular('MinFinalDistanceTipGoal', np.min([d[-1] for d in distances_tip_to_lego]))
        logger.record_tabular('MinDistanceTipGoal', np.mean([np.min(d) for d in distances_tip_to_lego]))

        distances_to_goal = [path["env_infos"]["reward_tip"] for path in paths]
        logger.record_tabular('RewardFinalDistanceLegoTip', np.mean([r[-1] for r in distances_to_goal]))
        if any([(d < self.distance_thresh).any() for d in distances_to_goal]):

            steps_to_thresh = np.mean([np.argmax(np.array(d) < self.distance_thresh) for d in distances_to_goal if (d < self.distance_thresh).any()])
        else:
            steps_to_thresh = len(distances_to_goal[0]) + 1
        time_to_thresh = steps_to_thresh * self.frame_skip * self.model.opt.timestep
        logger.record_tabular('TimeToGoal', time_to_thresh)
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        logger.record_tabular('PathsWithinThresh', paths_within_thresh)
    #


