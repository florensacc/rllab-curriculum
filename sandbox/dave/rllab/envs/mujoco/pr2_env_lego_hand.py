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
import copy

np.set_printoptions(threshold=np.nan, linewidth=np.nan)


class Pr2EnvLego(MujocoEnv, Serializable):
    FILE = 'hand.xml'

    def __init__(
            self,
            goal_generator=None,
            lego_generator=None,
            action_penalty_weight=0.001,  # originally was 0.001 #there is one with 0.0005
            distance_thresh=0.01,  # 1 cm
            model='hand.xml',  # 'pr2_1arm.xml',
            max_action=float("inf"),
            allow_random_restarts=True,  # same position: True
            allow_random_vel_restarts=True,
            qvel_init_std=1,  # 0.01,
            pos_normal_sample=False,
            pos_normal_sample_std=0.01,
            action_limiter=FixedActionLimiter(),
            use_running_average_failure_rate=False,
            failure_rate_gamma=0.9,
            mean_failure_rate_init=1.0,
            offset=np.zeros(3),
            use_vision=False,
            use_depth=False,
            number_actions=1,
            dilate_time=1,
            crop=True,
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
        self.pos_normal_sample_std = pos_normal_sample_std
        self.mean_failure_rate = mean_failure_rate_init
        self.failure_rate_gamma = failure_rate_gamma
        self.use_running_average_failure_rate = use_running_average_failure_rate
        self.offset = offset
        self.distance_tip_lego_penalty_weight = 0.6  # 0.5 #0.4 #1  #0.1  #0.3
        self.angle_penalty_weight = 0.2
        # 0.2 #0.4 #0.5 #1 #0.05
        self.occlusion_weight = 0.0005  # 0.0005
        self.use_vision = use_vision
        self.use_depth = use_depth
        self.discount = 0.95
        self.depth = np.zeros([99, 99, 3])  # TODO: Hacky
        self.model = model
        self.position_controller = True
        self.roll_joints = [2, 4]
        self.action = np.zeros((7,))
        self.first_action = True
        self.number_actions = number_actions
        self.noise = 0.01
        self.beta = 0.1
        self.crop = crop
        self.discount_weights = 0.99
        self.lego_pos = np.array([0, 0, 0])
        self.t = 0
        self.stop = False
        self.use_baseline = False

        super(Pr2EnvLego, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        init_qpos = copy.copy(self.model.data.qpos)
        init_qpos[:7, 0] = np.array([1.2, -0.5, 2, -1.5, 0, 0, -0.25])
        self.init_qpos = init_qpos

    def set_model(self, model):
        self.__class__.FILE = model

    def get_current_obs(self):
        vec_to_goal = self.get_vec_to_goal()
        dim = self.model.data.qpos.shape[0]
        obs = np.concatenate([
            self.model.data.qpos.flat,
            # We do not need to explicitly include the goal                                  # since we already have the vec to the goal.
            self.model.data.qvel.flat,  # Do not include the velocity of the target (should be 0).
            self.get_vec_tip_to_lego(),
            vec_to_goal,
        ]).reshape(-1)
        return obs

    def get_tip_position(self):
        sites = self.model.data.site_xpos
        return self.model.data.site_xpos[0]

    def get_lego_position(self):
        # Return the position of the lego block
        return self.model.data.site_xpos[-1]

    def get_vec_to_goal(self):
        lego_position = self.get_lego_position()
        # Compute the distance to the goal
        vec_to_goal = lego_position - (self.goal + self.offset)
        return vec_to_goal

    def get_vec_tip_to_lego(self):
        tip_position = self.get_tip_position()
        lego_position = self.get_lego_position()
        vec_tip_to_lego = lego_position - tip_position
        return vec_tip_to_lego

    def get_cos_vecs(self):
        vec_tip_to_lego = self.get_vec_tip_to_lego()
        vec_to_goal = self.get_vec_to_goal()
        return np.dot(vec_to_goal, vec_tip_to_lego) / (
            np.linalg.norm(vec_to_goal[:2]) * np.linalg.norm(vec_tip_to_lego[:2]))

    def step(self, action):

        #action = np.zeros_like(action)

        if self.use_baseline:
            if not self.stop:
                self.t += 0.05
                lg = self.goal - self.lego_pos
                action = self.lego_pos + lg * (self.t - 0.2 / np.linalg.norm(lg))
                action = action[:2]
            else:
                action = self.model.data.qpos[:2, 0]

        vec_to_goal_previous = self.get_vec_to_goal()
        distance_tip_to_goal_previous = np.linalg.norm(vec_to_goal_previous)

        self.forward_dynamics(action)

        vec_tip_to_lego = self.get_vec_tip_to_lego()
        vec_to_goal = self.get_vec_to_goal()
        distance_to_goal = np.linalg.norm(vec_to_goal)
        distance_tip_to_lego = np.linalg.norm(vec_tip_to_lego)


        tip_position = self.get_tip_position()
        lego_position = self.get_lego_position()
        vec_tip_to_lego2 = lego_position - tip_position
        # logger.log("Tip position: " + str(tip_position))
        # logger.log("lego position: " + str(lego_position))
        # logger.log("Tip to lego: " + str(vec_tip_to_lego2))
        # logger.log("distance_tip_to_lego: " + str(distance_tip_to_lego))
        #

        if self.t >= 1.4:
            self.stop = True
        # print(self.t)


        # Penalize the robot for being far from the goal and for having the arm far from the lego.
        reward_dist = - distance_to_goal
        reward_tip = - self.distance_tip_lego_penalty_weight * distance_tip_to_lego
        #logger.log("reward_tip: " + str(reward_tip))

        cos_angle = self.get_cos_vecs()
        reward_angle = - self.angle_penalty_weight * cos_angle

        reward = reward_dist + reward_angle + reward_tip
        # reward = reward_tip
        # reward = reward_tip + 2
        state = self._state
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self.get_current_obs()

        # Viewer
        if self.use_vision:
            self.viewer.loop_once()

        return Step(ob, float(reward), done,  # not self.do_rand,
                    distance_to_goal=distance_to_goal,
                    distance_to_goal_x=vec_to_goal[0],
                    distance_to_goal_y=vec_to_goal[1],
                    distance_to_goal_z=vec_to_goal[2],
                    distance_tip_to_lego=distance_tip_to_lego,
                    reward_dist=reward_dist,
                    reward_tip=reward_tip,
                    reward_angle=reward_angle,
                    weight_angle=self.angle_penalty_weight,
                    weight_tip=self.distance_tip_lego_penalty_weight,
                    x_goal=self.goal[0],
                    y_goal=self.goal[1],
                    x_lego=self.get_lego_position()[0],
                    y_lego=self.get_lego_position()[1],
                    )

    @overrides
    def reset_mujoco(self, qpos=None, qvel=None):
        goal_dims = 3  # self.goal_dims
        lego_dims = 6
        self.t = 0
        self.stop = False
        # print(self.error/100)
        # self.error = np.zeros((7,))
        qpos = copy.copy(self.model.data.qpos)

        # Generate block position with fixed orientation.
        if self.allow_random_restarts or self.first_time:
            lego_position = self.get_lego_position()

            if self._lego_generator is not None:
                self.lego = self._lego_generator.generate_goal(lego_position)
                qpos[-goal_dims - lego_dims - 1:-goal_dims] = self.lego[:, None]
            else:
                # print("No lego generator!")
                qpos[-goal_dims - lego_dims - 1:-goal_dims] = np.array((0.6, 0.2, 0.5025, 1, 0, 0, 0))[:, None]
        else:
            # Use current position as new position.
            qpos = copy.copy(self.model.data.qpos)  # [:-goal_dims]
            lego_position = self.get_lego_position()

        # Generate block position + uniform random orientation.
        # if self._lego_generator is not None:
        #     self.lego = self._lego_generator.generate_goal(lego_position)
        #     # Randomly select block orientation.
        #     theta = np.random.uniform(0, 2 * np.pi)
        #     # Convert orientation to quaternion + noise.
        #     quat = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)]) +\
        #            np.random.multivariate_normal(np.zeros((4,)), np.eye(4) * 0.001)
        #     qpos[-goal_dims - lego_dims - 1:-goal_dims, 0] = np.concatenate([self.lego[:3], quat])
        # else:
        #     # print("No lego generator!")
        #     qpos[-goal_dims - lego_dims - 1:-goal_dims] = np.array((0.6, 0.2, 0.5025, 1, 0, 0, 0))[:, None]

        # Generate a new goal.
        if self._goal_generator is not None:
            self.goal = self._goal_generator.generate_goal(self.lego[:3])
            qpos[-goal_dims:] = self.goal[:goal_dims, None]
        else:
            print("No goal generator!")

        # Manually initialize the hand to be on the opposite side of the lego from the goal.
        self.lego_pos = self.lego[:3]
        lg = (self.goal - self.lego_pos)
        init_hand = self.lego_pos + lg * (self.t - 0.25 / np.linalg.norm(lg))
        qpos[:2, 0] = init_hand[:2]

        if self.allow_random_vel_restarts or self.first_time:
            qvel = self.init_qvel + np.random.normal(size=self.init_qvel.shape) * self.qvel_init_std
        elif qvel is None:
            qvel = np.array(self.model.data.qvel)

        # Set the velocity of the goal (the goal itself -
        # this is NOT the arm velocity at the goal position!) to 0.
        qvel[-goal_dims - lego_dims:] = 0
        # qpos[:7] = 0
        # The position of a free body has 7 components (3 space and 4 for quaternions)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

        if self._action_limiter is not None:
            self.action_limit = self._action_limiter.get_action_limit()

        self.first_time = False
        # Apply a force in the Lego block
        xfrc = np.zeros(self.model.data.xfrc_applied.shape)
        xfrc[-2, 2] = -0.981
        # xfrc[13, 2] = - 9.81 * 0.0917
        self.model.data.xfrc_applied = xfrc
        # stiffness = np.random.uniform(0,10)
        # self.model.data.jnt_stiffness = np.array([stiffness]*14)[:, None]
        # Viewer
        if self.viewer is None and self.use_vision:
            self.viewer = MjViewer(visible=True, go_fast=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()

            # Choose an action limit from the range of action limits.
            # self.action_limit = np.random.uniform(self.min_action_limit, self.max_action_limit)

    def update_env(self, env_vars):
        self._goal_generator = env_vars.goal_generator
        self._action_limiter = env_vars.action_limiter

    def update_failure_rate(self, paths):
        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        failure_rate = 1 - paths_within_thresh
        if self.use_running_average_failure_rate:
            self.mean_failure_rate = self.mean_failure_rate * self.failure_rate_gamma + failure_rate * (
            1 - self.failure_rate_gamma)
        else:
            # We don't want to be dependent on the initial failure rate, so just use a large batch size.
            self.mean_failure_rate = failure_rate

    def viewer_setup(self, is_bot=False):
        self.viewer.cam.lookat[0] = self.model.stat.center[0]
        self.viewer.cam.lookat[1] = self.model.stat.center[1]
        self.viewer.cam.lookat[2] = self.model.stat.center[2]
        # self.viewer.cam.distance = self.model.stat.extent * 1.5


    def get_mean_failure_rate(self):
        return self.mean_failure_rate

    @overrides
    def log_diagnostics(self, paths):
        actions = [path["actions"] for path in paths]
        logger.record_tabular('MeanAbsActions', np.mean(np.abs(actions)))
        logger.record_tabular('MaxAbsActions', np.max(np.abs(actions)))
        logger.record_tabular('StdAbsActions', np.std(np.abs(actions)))
        logger.record_tabular('StdActions', np.std(actions))
        logger.record_tabular('MeanActions', np.mean(actions))


        distances_tip_to_lego = [path["env_infos"]["distance_tip_to_lego"] for path in paths]
        logger.record_tabular('MinFinalDistanceTipLego', np.min([d[-1] for d in distances_tip_to_lego]))
        logger.record_tabular('MinDistanceTipLego', np.mean([np.min(d) for d in distances_tip_to_lego]))

        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        logger.record_tabular('MinDistanceToGoal', np.mean([np.min(d) for d in distances_to_goal]))
        logger.record_tabular('MinFinalDistanceToGoal', np.min([d[-1] for d in distances_to_goal]))
        logger.record_tabular('FinalDistanceToGoal', np.mean([d[-1] for d in distances_to_goal]))
        distances_to_goal = [path["env_infos"]["reward_dist"] for path in paths]
        logger.record_tabular('RewardDistanceLegoGoal', np.mean([np.sum(r) for r in distances_to_goal]))
        distances_to_goal = [path["env_infos"]["reward_tip"] for path in paths]
        logger.record_tabular('RewardDistanceLegoTip', np.mean([np.sum(r) for r in distances_to_goal]))
        distances_to_goal = [path["env_infos"]["reward_angle"] for path in paths]
        logger.record_tabular('RewardAngle', np.mean([np.sum(r) for r in distances_to_goal]))
        distances_to_goal = [path["env_infos"]["weight_angle"] for path in paths]
        logger.record_tabular('WeightAngle', np.mean([np.mean(r) for r in distances_to_goal]))
        distances_to_goal = [path["env_infos"]["weight_tip"] for path in paths]
        logger.record_tabular('WeightTip', np.mean([np.mean(r) for r in distances_to_goal]))
        # distances_to_goal = [path["env_infos"]["reward_occlusion"] for path in paths]
        # logger.record_tabular('RewardOcclusion', np.mean([np.sum(r) for r in distances_to_goal]))
        # error_position_x = [path["env_infos"]["error_position_x"] for path in paths]
        # logger.record_tabular('ErrorpositionX', np.max([np.max(e) for e in error_position_x]))
        # error_position_y = [path["env_infos"]["error_position_y"] for path in paths]
        # logger.record_tabular('ErrorpositionY', np.max([np.max(e) for e in error_position_y]))
        # error_position_z = [path["env_infos"]["error_position_z"] for path in paths]
        # logger.record_tabular('ErrorpositionZ', np.max([np.max(e) for e in error_position_z]))
        # The task is considered complete when we get within distance_thresh of the goal.
        # reached_goal_indices = np.where(distances_to_goal < distance_thresh)
        # if (distances_to_goal < distance_thresh).any():
        if any([(d < self.distance_thresh).any() for d in distances_to_goal]):
            # distance_to_thresh = np.argmin(distances_to_goal < distance_thresh)
            # np.mean([np.argmin(d < distance_thresh) for d in distances_to_goal if np.argmin(d < distance_thresh) > 0])
            steps_to_thresh = np.mean([np.argmax(np.array(d) < self.distance_thresh) for d in distances_to_goal if
                                       (d < self.distance_thresh).any()])
        else:
            steps_to_thresh = len(distances_to_goal[0]) + 1
        time_to_thresh = steps_to_thresh * self.frame_skip * self.model.opt.timestep
        logger.record_tabular('TimeToGoal', time_to_thresh)
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        logger.record_tabular('PathsWithinThresh', paths_within_thresh)


        # def __getstate__(self):
        #     d = super(Pr2EnvLego, self).__getstate__()
        #     d['_weight_angle'], d['_weight_tip'] = self.angle_penalty_weight, self.distance_tip_lego_penalty_weight
        #     return d
        #
        # def __setstate__(self, d):
        #     super(Pr2EnvLego, self).__setstate__(d)
        #     self.update_weights(d['_weight_angle'], d['_weight_tip'])
        #
        # def update_weights(self, angle_weight, tip_weight):
        #     self.angle_penalty_weight = angle_weight * 0.997
        #     self.distance_tip_lego_penalty_weight = tip_weight * 0.997
        #     # return self.angle_penalty_weight, self.distance_tip_lego_penalty_weight

        # #
