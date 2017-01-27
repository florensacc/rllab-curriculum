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

        super(Pr2EnvLego, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def set_model(self, model):
        self.__class__.FILE = model

    def get_current_obs(self):
        vec_to_goal = self.get_vec_to_goal()
        dim = self.model.data.qpos.shape[0]

        if self.use_depth:
            self.depth = self.viewer.get_depth_map()
            # depth = self.depth.astype(np.uint8)
            # depth = imresize(depth, (227, 227))
            # depth = depth.astype(np.float32).transpose([2, 0, 1])
            # img,_,_ = self.viewer.get_image()
            # img = img.astype(np.uint8)
            # imsave('data/local/imgs/lego' + str(time.clock()) + '.png', img)
            # depth = self.depth.transpose([2,0,1])
            idxpos = list(range(7)) + list(range(dim - 3, dim))  # TODO: Hacky
            idxvel = list(range(7))
            return  np.concatenate([
                self.model.data.qpos.flat[idxpos],
                # self.model.data.qpos.flat[:-3], # We do not need to explicitly include the goal
                #                                 # since we already have the vec to the goal.
                self.model.data.qvel.flat[idxvel],  # Do not include the velocity of the target (should be 0).
                self.get_tip_position(),
                # self.get_vec_tip_to_lego(),
                np.reshape(self.depth, -1),
            ]).reshape(-1)
        else:
            idxpos = list(range(7)) + list(range(14, dim))  # TODO: Hacky
            idxvel = list(range(7)) + list(range(14, dim - 3 - 1))
            return np.concatenate([
                self.model.data.qpos.flat[idxpos],  # We do not need to explicitly include the goal
                #                                  # since we already have the vec to the goal.
                self.model.data.qvel.flat[idxvel],  # Do not include the velocity of the target (should be 0).
                self.get_tip_position(),
                self.get_vec_tip_to_lego(),
                vec_to_goal,
            ]).reshape(-1)

    def get_tip_position(self):
        #return self.get_body_com("l_gripper_r_finger_tip_link")
        return self.model.data.site_xpos[0]

    def get_lego_position(self):
        #Return the position of the lego block
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
        return np.dot(vec_to_goal[:2], vec_tip_to_lego[:2]) / (
            np.linalg.norm(vec_to_goal[:2]) * np.linalg.norm(vec_tip_to_lego[:2]))


    # def get_reward_occlusion(self):
    #     _image, height, width = self.viewer.get_image()
    #     image = np.uint8(_image)[40:70, 10:80, :].copy()
    #     lower_boundary_red = np.array([130, 0, 0], dtype='uint8')
    #     upper_boundary_red = np.array([255, 90, 90], dtype='uint8')
    #     mask_red = cv2.inRange(image, lower_boundary_red, upper_boundary_red)
    #     return np.sum(mask_red.astype(np.float32)) / 255.
    #
    # def get_lego_from_image(self):
    #     _image, height, width = self.viewer.get_image()
    #     h_lego = 11
    #     w_lego = 7
    #     # image = image[195:360,50:450,:]
    #     # _image = cv2.resize(_image, (4 * 99, 4 * 99))
    #     # image = cv2.cvtColor(np.uint8(_image)[70:40:-1, 10:80,:], cv2.COLOR_RGB2HSV)
    #     image = np.uint8(_image)[40:70, 10:80,:]
    #     # mask_red = cv2.inRange(image, np.array([0, 200, 130]), np.array([1, 255, 255]))
    #     # mask_red = mask1_red
    #     lower_boundary_red = np.array([130, 30, 30], dtype='uint8')
    #     upper_boundary_red = np.array([255, 90, 90], dtype='uint8')
    #     # lower_boundary_green = np.array([5, 60, 5], dtype='uint8')
    #     # upper_boundary_green = np.array([65, 255, 65], dtype='uint8')
    #     mask_red = cv2.inRange(image, lower_boundary_red, upper_boundary_red)
    #     # mask_red = cv2.resize(mask_red, (2*210, 2*90))
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #     mask_green = cv2.inRange(image, np.array([55, 0, 0]), np.array([65, 255, 255]))
    #     mask2_green = cv2.inRange(image, np.array([254, -1, -1]), np.array([256, 2, 2]))
    #     # mask_green = (mask1_green + mask2_green)/255.
    #     corners = cv2.goodFeaturesToTrack(mask_red, 10, 1e-4, 4)
    #     filter_size = 3
    #     lego_corners = []
    #     thresh = 16 * 255
    #     # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    #     if corners is not None:
    #         mask_green= cv2.copyMakeBorder(mask_green,filter_size, filter_size, filter_size,
    #                                       filter_size, cv2.BORDER_CONSTANT, value=0)
    #         corners = np.int0(corners)
    #         for corner in corners:
    #             x, y = corner.ravel()
    #             greenpix = np.sum(mask_green[y:y + 2*filter_size + 1, x:x + 2*filter_size + 1])
    #             print(greenpix)
    #
    #             if greenpix > thresh:
    #                 lego_corners.append((40+y, 10+x))
    #                 cv2.circle(image, (x, y), 4, 255, -1)
    #             # image[y,x,0] = 255
    #
    #     # pdb.set_trace()
    #     print('len', len(lego_corners))
    #     lego_corners.sort()
    #     if len(lego_corners) == 4:
    #         #No oclusions for now, lego block always perpendicular to the table
    #         world_coordinates = []
    #         for r, c in lego_corners:
    #             world_coordinates.append(self.viewer.get_3D(r, c, self.depth[0, r, c]))
    #         lego_position = np.mean(world_coordinates, 0)
    #         uw = world_coordinates[1] - world_coordinates[0]
    #         uh = world_coordinates[2] - world_coordinates[0]
    #         uw_norm = np.linalg.norm(uw)
    #         uh_norm = np.linalg.norm(uh)
    #         if uw_norm > uh_norm:
    #             uw, uh = uh, uw
    #         cos_theta = np.dot(uh,np.array([1,0,0]))/uh_norm
    #         cos_half_theta = np.sqrt(0.5*(1+cos_theta)) * np.sign(cos_theta)
    #         sin_half_theta = np.sqrt(0.5*(1-cos_theta)) * np.sign(cos_theta)
    #         quaternion = np.array([cos_half_theta, 0, 0, -sin_half_theta])
    #     if len(lego_corners) != 4 and len(lego_corners) != 2 and len(lego_corners) != 3:
    #         # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    #         # cv2.imshow('i', image)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #         # import pdb; pdb.set_trace()
    #         return np.zeros(7)
    #
    #     elif len(lego_corners) == 3:
    #         world_coordinates = []
    #         for r, c in lego_corners:
    #             world_coordinates.append(self.viewer.get_3D(r, c, self.depth[0, r, c]))
    #             world_coordinates[-1][-1] = 0.5015
    #         us = [world_coordinates[2] - world_coordinates[1],
    #               world_coordinates[2] - world_coordinates[0],
    #               world_coordinates[1] - world_coordinates[0]]
    #         us_norm = np.sqrt(np.sum(np.square(us), axis=-1)).tolist()
    #         index = np.argmax(us_norm) #it is done that the index is the corner
    #         uw, uh = us[:index] + us[index+1:]
    #         uw_norm, uh_norm = us_norm[:index] + us_norm[index+1:]
    #         sign = 1
    #         if uw_norm > uh_norm:
    #             uw, uh = uh, uw
    #             uw_norm, uh_norm = uh_norm, uw_norm
    #             sign = -1
    #
    #         cos_theta = np.dot(uh, np.array([1, 0, 0])) / uh_norm
    #         cos_half_theta = np.sqrt(0.5 * (1 + cos_theta)) * np.sign(cos_theta)
    #         sin_half_theta = np.sqrt(0.5 * (1 - cos_theta)) * np.sign(cos_theta)
    #         quaternion = np.array([cos_half_theta, 0, 0, -sin_half_theta])
    #         if index == 0:
    #             lego_position = world_coordinates[index] + (uw/uw_norm * 0.0475/2 + uh/uh_norm * 0.0975)
    #         elif index == 1:
    #             lego_position = world_coordinates[index] + sign * (uw/uw_norm * 0.0475/2 - uh/uh_norm * 0.0975)
    #         else:
    #             lego_position = world_coordinates[index] - (uw/uw_norm * 0.0475/2 + uh/uh_norm * 0.0975)
    #
    #     if len(lego_corners) == 2:
    #         world_coordinates = []
    #         for r, c in lego_corners:
    #             world_coordinates.append(self.viewer.get_3D(r, c, self.depth[0, r, c]))
    #         u = world_coordinates[1] - world_coordinates[0]
    #         if np.linalg.norm(u) + 0.06 > 0.0975 * 2:
    #             u[-1] = 0
    #             uh = u
    #             uw = np.array([-u[1], u[0], u[2]])
    #             uh_image = np.array(lego_corners[1]) - np.array(lego_corners[0])
    #             uw_image = np.array([uh_image[1], -uh_image[0]])
    #             p = np.array(lego_corners[0]) + uh_image/np.linalg.norm(uh_image) * filter_size + uw_image/np.linalg.norm(uw_image) * filter_size
    #             if np.sum(mask_green[p[1]:p[1] + 2*filter_size + 1, p[0]:p[0] + 2*filter_size + 1]) > thresh:
    #                 uw = -uw
    #             # lego_position = world_coordinates[0] + uh/2+ uw/np.linalg.norm(uw) * 0.0475 / 2 #lego width
    #             lego_position = world_coordinates[0] + uh/np.linalg.norm(uh)*0.0975+ uw/np.linalg.norm(uw) * 0.0475 / 2 #lego width
    #             cos_theta = np.dot(uh, np.array([1, 0, 0])) / np.linalg.norm(uh)
    #             cos_half_theta = np.sqrt(0.5 * (1 + cos_theta)) * np.sign(cos_theta)
    #             sin_half_theta = np.sqrt(0.5 * (1 - cos_theta)) * np.sign(cos_theta)
    #             quaternion = np.array([cos_half_theta, 0, 0, -sin_half_theta])
    #         else:
    #             u[-1] = 0
    #             uw = u
    #             uh = np.array([u[1], -u[0], u[2]])  #TODO: Not sure about the signs
    #             uw_image = np.array(lego_corners[1]) - np.array(lego_corners[0])
    #             uh_image = np.array([uw_image[1], -uw_image[0]])
    #             p = np.array(lego_corners[0]) + uh_image / np.linalg.norm(uh_image) * 2 + uw_image / np.linalg.norm(
    #                 uw_image) * 2
    #             if np.sum(mask_green[p[1]:p[1] + 2 * filter_size + 1, p[0]:p[0] + 2 * filter_size + 1]) > thresh:
    #                 uh = -uh
    #             lego_position = world_coordinates[0] + uw / 2 + uh/np.linalg.norm(uh) * 0.0975  # lego width
    #
    #             cos_theta = np.dot(uh, np.array([1, 0, 0])) / np.linalg.norm(uh)
    #             cos_half_theta = np.sqrt(0.5 * (1 + cos_theta)) * np.sign(cos_theta)
    #             sin_half_theta = np.sqrt(0.5 * (1 - cos_theta)) * np.sign(cos_theta)
    #             quaternion = np.array([cos_half_theta, 0, 0, -sin_half_theta])
    #
    #
    #
    #
    #     lego_position[-1] = 0.4815
    #     error_position = np.linalg.norm(lego_position - self.get_lego_position())
    #     print(error_position, lego_position, self.get_lego_position())
    #     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    #     # cv2.imshow('i', image); cv2.waitKey(0); cv2.destroyAllWindows()
    #     # cv2.waitKey(5)
    #     # cv2.destroyAllWindows()
    #     import pdb; pdb.set_trace()
    #     return np.concatenate([lego_position, quaternion])

    def step(self, action):
        # Limit actions to the specified range.
        if self.use_depth and self.use_vision:
            self.depth = self.viewer.get_depth_map().astype(np.float32)
        action_limit = self.action_limit * self.action_space.ones()
        action = np.maximum(action, -action_limit)
        action = np.minimum(action, action_limit)
        vec_tip_to_lego = self.get_vec_tip_to_lego()
        distance_tip_to_lego_previous = np.linalg.norm(vec_tip_to_lego)
        reward_tip_previous = - self.distance_tip_lego_penalty_weight * distance_tip_to_lego_previous
        cos_angle_previous = self.get_cos_vecs()
        reward_angle_previous = - self.angle_penalty_weight * cos_angle_previous

        # import pdb; pdb.set_trace()
        # action = np.zeros_like(action)
        self.forward_dynamics(action)

        # Simulate this action and get the resulting state.
        # image_pos = np.zeros(3)
        # a = self.get_lego_from_image()
        # if a is not None:
        #     image_pos = a

        vec_to_goal = self.get_vec_to_goal()
        distance_to_goal = np.linalg.norm(vec_to_goal)
        distance_tip_to_lego = np.linalg.norm(vec_tip_to_lego)

        # print("calculat:  ", self.get_lego_from_image(), "|", "real:  ", self.get_lego_position())
        # Penalize the robot for being far from the goal and for having the arm far from the lego.
        reward_dist = - distance_to_goal
        reward_tip = - self.distance_tip_lego_penalty_weight * distance_tip_to_lego

        cos_angle = self.get_cos_vecs()
        reward_angle = - self.angle_penalty_weight * cos_angle

        # Penalize the robot for large actions.f
        # reward_occlusion = self.occlusion_weight * self.get_reward_occlusion()
        reward_ctrl = - self.action_penalty_weight * np.square(action).sum()
        reward = reward_dist + reward_ctrl + reward_tip + reward_angle
        state = self._state
        # print(reward_occlusion, reward_angle, reward_tip, reward_dist, )
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self.get_current_obs()


        # Viewer
        if self.use_vision:
            self.viewer.loop_once()

        return Step(ob, float(reward), done, #not self.do_rand,
                    distance_to_goal=distance_to_goal,
                    distance_to_goal_x=vec_to_goal[0],
                    distance_to_goal_y=vec_to_goal[1],
                    distance_to_goal_z=vec_to_goal[2],
                    distance_tip_to_lego=distance_tip_to_lego,
                    reward_dist=reward_dist,
                    reward_tip=reward_tip,
                    reward_angle=reward_angle,
                    # reward_occlusion=reward_occlusion,
                    # error_position_x=error_position[0],
                    # error_position_y=error_position[1],
                    # error_position_z=error_position[2],
                    )

    def viewer_setup(self, is_bot=False):
        #self.viewer.cam.lookat[0] = self.model.stat.center[0]
        #self.viewer.cam.lookat[1] = self.model.stat.center[1]
        #self.viewer.cam.lookat[2] = self.model.stat.center[2]
        if self.use_vision:
            self.viewer.cam.camid = -1
            # self.viewer.cam.distance = self.model.stat.extent * 1.5
        else:
            self.viewer.cam.camid = -1
            self.viewer.cam.distance = self.model.stat.extent * 1.5
        #self.viewer.cam.trackbodyid = -1   # 39
        #self.viewer.cam.elevation = 0
        #self.viewer.cam.azimuth = 0
        #self.viewer.cam.VR = 1


    @overrides
    def reset_mujoco(self, qpos=None, qvel=None):
        goal_dims = 3 # self.goal_dims
        lego_dims = 6
        self.first_time = True
        if self.allow_random_restarts or self.first_time:
            if self.pos_normal_sample:
                # Sample a new random initial robot position from a normal distribution.
                # ini = np.zeros(self.init_qpos.shape)
                # ini[0:3] = np.reshape(np.array([0.5, 0, 6]), (3,1))
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
                    if idx == 1:
                        qpos[idx] = max(jnt_range[0], qpos[idx])
                        qpos[idx] = min(jnt_range[0]*0.25 + jnt_range[1]*0.75, qpos[idx])
                    if idx == 3:
                        qpos[idx] = max(jnt_range[0], qpos[idx])
                        qpos[idx] = min(jnt_range[0] * 0.3 + 0.7 * jnt_range[1], qpos[idx])

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

        self.init_qpos = qpos
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

        if self._action_limiter is not None:
            self.action_limit = self._action_limiter.get_action_limit()

        # self.first_time = False

        xfrc = np.zeros(self.model.data.xfrc_applied.shape)
        xfrc[-2, 2] = -0.981
        xfrc[13, 2] = - 9.81 * 0.0917
        self.model.data.xfrc_applied = xfrc
        import copy
        damping = copy.copy(self.model.dof_damping)[:,0]
        armature = copy.copy(self.model.dof_armature )[:, 0]
        frictionloss = copy.copy(self.model.dof_frictionloss)[:,0]
        dim = len(damping)
        damping = np.maximum(0, np.random.multivariate_normal(damping, 0.001 * np.eye(dim)))
        armature = np.maximum(0, np.random.multivariate_normal(armature, 0.001 * np.eye(dim)))
        frictionloss = np.maximum(0, np.random.multivariate_normal(frictionloss, 0.001 * np.eye(dim)))
        self.model.dof_damping = damping[:, None]
        self.model.dof_frictionloss = frictionloss[:, None]
        self.model.dof_armature = armature[:, None]
        # #Viewer



        if self.viewer is None and self.use_vision:
            self.viewer = MjViewer(visible=True, go_fast=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()

        self.first_time = False

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

        #distances_to_goal_x = [path["env_infos"]["distance_to_goal_x"] for path in paths]
        #distances_to_goal_y = [path["env_infos"]["distance_to_goal_y"] for path in paths]
        #distances_to_goal_z = [path["env_infos"]["distance_to_goal_z"] for path in paths]

        #logger.record_tabular('FinalDistanceToGoalX', np.mean([d[-1] for d in distances_to_goal_x]))
        #logger.record_tabular('FinalDistanceToGoalY', np.mean([d[-1] for d in distances_to_goal_y]))
        #logger.record_tabular('FinalDistanceToGoalZ', np.mean([d[-1] for d in distances_to_goal_z]))


        # logger.record_tabular('MaxFinalDistanceToGoalX', np.max([d[-1] for d in distances_to_goal_x]))
        # logger.record_tabular('MaxFinalDistanceToGoalY', np.max([d[-1] for d in distances_to_goal_y]))
        # logger.record_tabular('MaxFinalDistanceToGoalZ', np.max([d[-1] for d in distances_to_goal_z]))
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
        # distances_to_goal = [path["env_infos"]["reward_occlusion"] for path in paths]
        # logger.record_tabular('RewardOcclusion', np.mean([np.sum(r) for r in distances_to_goal]))
        # error_position_x = [path["env_infos"]["error_position_x"] for path in paths]
        # logger.record_tabular('ErrorpositionX', np.max([np.max(e) for e in error_position_x]))
        # error_position_y = [path["env_infos"]["error_position_y"] for path in paths]
        # logger.record_tabular('ErrorpositionY', np.max([np.max(e) for e in error_position_y]))
        # error_position_z = [path["env_infos"]["error_position_z"] for path in paths]
        # logger.record_tabular('ErrorpositionZ', np.max([np.max(e) for e in error_position_z]))
        # The task is considered complete when we get within distance_thresh of the goal.
        #reached_goal_indices = np.where(distances_to_goal < distance_thresh)
        #if (distances_to_goal < distance_thresh).any():
        if any([(d < self.distance_thresh).any() for d in distances_to_goal]):
            #distance_to_thresh = np.argmin(distances_to_goal < distance_thresh)
            #np.mean([np.argmin(d < distance_thresh) for d in distances_to_goal if np.argmin(d < distance_thresh) > 0])
            steps_to_thresh = np.mean([np.argmax(np.array(d) < self.distance_thresh) for d in distances_to_goal if (d < self.distance_thresh).any()])
        else:
            steps_to_thresh = len(distances_to_goal[0]) + 1
        time_to_thresh = steps_to_thresh * self.frame_skip * self.model.opt.timestep
        logger.record_tabular('TimeToGoal', time_to_thresh)
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        logger.record_tabular('PathsWithinThresh', paths_within_thresh)

        # pos_dim = len(self.model.data.qpos.flat)
        # vel_dim = len(self.model.data.qvel.flat[:-3])
        # Timesteps,
        # observations = [path["observations"] for path in paths]
        #velocities = [path["observations"][:][pos_dim+1 : pos_dim+vel_dim] for path in paths]
        # velocities_nested = observations[:][pos_dim+1 : pos_dim+vel_dim]
        # velocities = list(itertools.chain.from_iterable(velocities_nested))
        # logger.record_tabular("MeanVelocities", np.mean(np.abs(velocities)))
        # logger.record_tabular("MaxVelocities", np.max(np.abs(velocities)))
        # print "Mean vel: " + str(np.mean(np.mean(np.abs(velocities), 1),1))
        # print "Max vel: " + str(np.max(np.max(np.abs(velocities), 1),1))

        # goal_generator_diagnostics = self._goal_generator.get_diagnostics()
        # for key, val in goal_generator_diagnostics.items():
        #   logger.record_tabular(key, val)

        # action_limiter_diagnostics = self._action_limiter.get_diagnostics()
        # for key, val in action_limiter_diagnostics.items():
        #    logger.record_tabular(key, val)

        # self.update_failure_rate(paths)

        # action_limit = self._action_limiter.get_action_limit()
        # failure_rate = self.get_mean_failure_rate()
        # expected_damage = action_limit * failure_rate
        # logger.record_tabular('Expected Damage', expected_damage)


