



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

    FILE = 'pr2_lego_calibration.xml' #'pr2_1arm.xml'

    def __init__(
            self,
            goal_generator=None,
            lego_generator=None,
            action_penalty_weight= 0.001, #originally was 0.001 #there is one with 0.0005
            distance_thresh=0.01,  # 1 cm
            model='pr2_lego_calibration.xml', #'pr2_1arm.xml',
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
        self._action_limiter = action_limiter
        self.allow_random_restarts = allow_random_restarts
        self.allow_random_vel_restarts = allow_random_vel_restarts
        self.first_time = True
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

        super(Pr2EnvLego, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def set_model(self, model):
        self.__class__.FILE = model

    # def get_tip_position(self):
    #     return self.model.data.site_xpos[0]
    #
    # def get_goal_position(self):
    #     return self.model.data.site_xpos[-1]
    #
    # def get_vec_goal_to_tip(self):
    #     ee_pos = self.get_tip_position()
    #     goal_pos = self.get_goal_position()
    #     return ee_pos - goal_pos

    def get_current_obs(self):
        # ee_pos = self.get_tip_position()
        # goal_pos = self.get_goal_position()
        idx = list(range(7))
        return np.concatenate([
            self.model.data.qpos.flat[idx],  # We do not need to explicitly include the goal
            self.model.data.qvel.flat[idx],  # Do not include the velocity of the target (should be 0).
            # ee_pos,
            # goal_pos,
        ]).reshape(-1)

    def step(self, action):
        #action /= 10

        # Limit actions to the specified range.

        if self.first_time:
            self.dilate_time = 1000
            ctrl = np.zeros_like(self.init_ctrl)
            self.forward_dynamics(ctrl, qvel=self.init_qvel, qpos=self.init_qpos)
            self.first_time = False
            self.dilate_time = 1

        # print(self.model.data.qacc)

        action_limit = self.action_limit * self.action_space.ones()
        action = np.maximum(action, -action_limit)
        action = np.minimum(action, action_limit)
        # action = np.zeros_like(action)
        self.forward_dynamics(action)
        # vec = self.get_vec_goal_to_tip()
        # reward =  -np.linalg.norm(vec)
        reward = 1
        state = self._state
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self.get_current_obs()

        # Viewer
        if self.use_vision:
            self.viewer.loop_once()

        return Step(ob, float(reward), done, #not self.do_rand,
                    )

    def viewer_setup(self, is_bot=False):
        self.viewer.cam.camid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.5



    @overrides
    def reset_mujoco(self, qpos=None, qvel=None):

        import copy
        joint_idx = 3
        if qpos is None:
            # Use current position as new position.
            qpos_curr = self.model.data.qpos #[:-goal_dims]
            self.init_qpos = list(qpos_curr)
        else:
            qpos_curr = copy.copy(self.model.data.qpos)
            qpos_curr[:7] = qpos[:, None]
            self.init_qpos = qpos_curr
            # self.model.data.qpos = qpos_curr

        jnt_limited = copy.copy(self.model.jnt_limited)
        range = copy.copy(self.model.jnt_range)

        # import pdb; pdb.set_trace()
        for idx, jnt_range in enumerate(self.model.jnt_range):
            if self.model.jnt_limited[idx] == 1:
                self.init_qpos[idx] = max(jnt_range[0], self.init_qpos[idx])
                self.init_qpos[idx] = min(jnt_range[1], self.init_qpos[idx])



        xfrc = np.zeros(self.model.data.xfrc_applied.shape)
        xfrc[13, 2] = - 9.81 * 0.0917
        self.model.data.xfrc_applied = xfrc

        if qvel is None:
            qvel = np.array(self.model.data.qvel)
        else:
            qvel_curr = copy.copy(self.model.data.qvel)
            qvel_curr[:7] = qvel[:, None]
            self.init_qvel = qvel_curr
            self.model.data.qvel = qvel_curr




        #The position of a free body has 7 components (3 space and 4 for quaternions)

        # import pdb; pdb.set_trace()

        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        # self.model.data.qacc = np.zeros_like(self.model.data.qacc)

        # joint_idx = 0
        # import copy
        # #
        damping = copy.copy(self.model.dof_damping)
        stiffness = copy.copy(self.model.jnt_stiffness)
        armature = copy.copy(self.model.dof_armature)
        frictionloss = copy.copy(self.model.dof_frictionloss)
        # inertia = copy.copy(self.model.body_inertia)
        # import pdb; pdb.set_trace()


        # damping[joint_idx] = 0.5
        # stiffness[joint_idx] = 1.5
        # armature[joint_idx] = 100
        # frictionloss[joint_idx] = 0.1
        # inertia[joint_idx + 3] = 0.1
        cal_params = {
        # 'l_wrist_roll_joint': {'joint_damping': 0.0024830019711590584, 'joint_frictionloss': 2.1086307637454137e-39,
        #                         'joint_armature': 0.00085619352737954476, 'joint_stiffness': 2.1604387432122759e-11},
         'l_shoulder_pan_joint': {'joint_damping': 0.00345240094517906, 'joint_frictionloss': 0.45209647293908345,
                                  'joint_armature': 0.0013905360987456782, 'joint_stiffness': 0.078326399298161986},
         # 'l_upper_arm_roll_joint': {'joint_damping': 8.3058605903160154e-05, 'joint_frictionloss': 0.063461842233377599,
         #                            'joint_armature': 0.0014367825464401203, 'joint_stiffness': 0.0663322179824785},
         'l_shoulder_lift_joint': {'joint_damping': 0.023440485143310975, 'joint_frictionloss': 0.6005660273687522,
                                   'joint_armature': 0.0055501186980643049, 'joint_stiffness': 0.34671347187675627},
         'l_elbow_flex_joint': {'joint_damping': 0.010386442811917787, 'joint_frictionloss': 3.2869682973475361e-05,
                                'joint_armature': 0.0010511307853910354, 'joint_stiffness': 0.029568831082159765},
         # 'l_forearm_roll_joint': {'joint_damping': 0.0019368158670266891, 'joint_frictionloss': 2.9793032369557001e-17,
         #                          'joint_armature': 0.00088303013294424636, 'joint_stiffness': 6.1726733960484178e-17},
         'l_wrist_flex_joint': {'joint_damping': 0.01289360720339546, 'joint_frictionloss': 4.0478683737576653e-08,
                                'joint_armature': 0.0015341977055686928, 'joint_stiffness': 0.040413314700643854}}

        params = [
            'joint_damping',
            'joint_frictionloss',
            'joint_armature',

        ]
        joints = {"l_shoulder_pan_joint":0, \
                  "l_shoulder_lift_joint":1, \
                  "l_upper_arm_roll_joint":2, \
                  "l_elbow_flex_joint":3, \
                  "l_forearm_roll_joint":4, \
                  "l_wrist_flex_joint":5, \
                  "l_wrist_roll_joint":6, \
                  }

        for joint in cal_params.keys():
            joint_idx = joints[joint]
            for param in cal_params[joint].keys():
                if param == 'joint_damping':
                    damping[joint_idx] = cal_params[joint][param]
                if param == 'joint_frictionloss':
                    frictionloss[joint_idx] = cal_params[joint][param]
                if param == 'joint_armature':
                    armature[joint_idx] = cal_params[joint][param]

        self.model.dof_damping = damping * 200
        self.model.dof_frictionloss = frictionloss * 1
        self.model.dof_armature = armature * 200
        # import pdb; pdb.set_trace()



############ DONE ##################

        # joint_idx = 0
        # damping[joint_idx] = 0.5
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 100
        # stiffness[joint_idx] = 1.5
        # #
        # joint_idx = 1
        # damping[joint_idx] = 0.3
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 5
        # stiffness[joint_idx] = 5.5
        # #
        # joint_idx = 2
        # damping[joint_idx] = .07
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 0.3
        # stiffness[joint_idx] = 0.65
        # #
        # joint_idx = 3
        # damping[joint_idx] = 1.5
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 0.7
        # stiffness[joint_idx] = 0.27
        # #
        # joint_idx = 4
        # damping[joint_idx] = 0.55
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 0.005
        # stiffness[joint_idx] = 0.035
        # #
        # joint_idx = 5
        # damping[joint_idx] = 100
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 100
        # stiffness[joint_idx] = 0.1
        # #
        # joint_idx = 6
        # damping[joint_idx] = 0.39
        # frictionloss[joint_idx] = 0.1
        # armature[joint_idx] = 0.05
        # stiffness[joint_idx] = 0.01


        # damping[joint_idx] = 0.8
        # stiffness[joint_idx] = 0.5
        # armature[joint_idx] = 0.8
        # frictionloss[joint_idx] = 0.7
        # inertia[joint_idx + 3] = 0.1

        self.model.jnt_stiffness = stiffness
        self.model.dof_damping = damping
        self.model.dof_armature = armature
        self.dof_frictionloss = frictionloss
        # self.model.body_inertia = inertia

        if self._action_limiter is not None:
            self.action_limit = self._action_limiter.get_action_limit()


        #Viewer

        if self.viewer is None and self.use_vision:
            self.viewer = MjViewer(visible=True, go_fast=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()


        # Choose an action limit from the range of action limits.
        #self.action_limit = np.random.uniform(self.min_action_limit, self.max_action_limit)

    def update_env(self, env_vars):
        self._action_limiter = env_vars.action_limiter

    @overrides
    def log_diagnostics(self, paths):
        actions = [path["actions"] for path in paths]
        logger.record_tabular('MeanAbsActions', np.mean(np.abs(actions)))
        logger.record_tabular('MaxAbsActions', np.max(np.abs(actions)))
        logger.record_tabular('StdAbsActions', np.std(np.abs(actions)))
        logger.record_tabular('StdActions', np.std(actions))

        #distances_to_goal_x = [path["env_infos"]["distance_to_goal_x"] for path in paths]
        #distances_to_goal_y = [path["env_infos"]["distance_to_goal_y"] for path in paths]
        # #distances_to_goal_z = [path["env_infos"]["distance_to_goal_z"] for path in paths]
        #
        # #logger.record_tabular('FinalDistanceToGoalX', np.mean([d[-1] for d in distances_to_goal_x]))
        # #logger.record_tabular('FinalDistanceToGoalY', np.mean([d[-1] for d in distances_to_goal_y]))
        # #logger.record_tabular('FinalDistanceToGoalZ', np.mean([d[-1] for d in distances_to_goal_z]))
        #
        #
        # # logger.record_tabular('MaxFinalDistanceToGoalX', np.max([d[-1] for d in distances_to_goal_x]))
        # # logger.record_tabular('MaxFinalDistanceToGoalY', np.max([d[-1] for d in distances_to_goal_y]))
        # # logger.record_tabular('MaxFinalDistanceToGoalZ', np.max([d[-1] for d in distances_to_goal_z]))
        # distances_tip_to_lego = [path["env_infos"]["distance_tip_to_lego"] for path in paths]
        # logger.record_tabular('MinFinalDistanceTipLego', np.min([d[-1] for d in distances_tip_to_lego]))
        # logger.record_tabular('MinDistanceTipLego', np.mean([np.min(d) for d in distances_tip_to_lego]))
        #
        # distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        # logger.record_tabular('MinDistanceToGoal', np.mean([np.min(d) for d in distances_to_goal]))
        # logger.record_tabular('MinFinalDistanceToGoal', np.min([d[-1] for d in distances_to_goal]))
        # logger.record_tabular('FinalDistanceToGoal', np.mean([d[-1] for d in distances_to_goal]))
        # distances_to_goal = [path["env_infos"]["reward_dist"] for path in paths]
        # logger.record_tabular('RewardDistanceLegoGoal', np.mean([np.sum(r) for r in distances_to_goal]))
        # distances_to_goal = [path["env_infos"]["reward_tip"] for path in paths]
        # logger.record_tabular('RewardDistanceLegoTip', np.mean([np.sum(r) for r in distances_to_goal]))
        # distances_to_goal = [path["env_infos"]["reward_angle"] for path in paths]
        # logger.record_tabular('RewardAngle', np.mean([np.sum(r) for r in distances_to_goal]))
        # # distances_to_goal = [path["env_infos"]["reward_occlusion"] for path in paths]
        # # logger.record_tabular('RewardOcclusion', np.mean([np.sum(r) for r in distances_to_goal]))
        # # error_position_x = [path["env_infos"]["error_position_x"] for path in paths]
        # # logger.record_tabular('ErrorpositionX', np.max([np.max(e) for e in error_position_x]))
        # # error_position_y = [path["env_infos"]["error_position_y"] for path in paths]
        # # logger.record_tabular('ErrorpositionY', np.max([np.max(e) for e in error_position_y]))
        # # error_position_z = [path["env_infos"]["error_position_z"] for path in paths]
        # # logger.record_tabular('ErrorpositionZ', np.max([np.max(e) for e in error_position_z]))
        # # The task is considered complete when we get within distance_thresh of the goal.
        # #reached_goal_indices = np.where(distances_to_goal < distance_thresh)
        # #if (distances_to_goal < distance_thresh).any():
        # if any([(d < self.distance_thresh).any() for d in distances_to_goal]):
        #     #distance_to_thresh = np.argmin(distances_to_goal < distance_thresh)
        #     #np.mean([np.argmin(d < distance_thresh) for d in distances_to_goal if np.argmin(d < distance_thresh) > 0])
        #     steps_to_thresh = np.mean([np.argmax(np.array(d) < self.distance_thresh) for d in distances_to_goal if (d < self.distance_thresh).any()])
        # else:
        #     steps_to_thresh = len(distances_to_goal[0]) + 1
        # time_to_thresh = steps_to_thresh * self.frame_skip * self.model.opt.timestep
        # logger.record_tabular('TimeToGoal', time_to_thresh)
        # paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        # logger.record_tabular('PathsWithinThresh', paths_within_thresh)
        #
        # # pos_dim = len(self.model.data.qpos.flat)
        # # vel_dim = len(self.model.data.qvel.flat[:-3])
        # # Timesteps,
        # # observations = [path["observations"] for path in paths]
        # #velocities = [path["observations"][:][pos_dim+1 : pos_dim+vel_dim] for path in paths]
        # # velocities_nested = observations[:][pos_dim+1 : pos_dim+vel_dim]
        # # velocities = list(itertools.chain.from_iterable(velocities_nested))
        # # logger.record_tabular("MeanVelocities", np.mean(np.abs(velocities)))
        # # logger.record_tabular("MaxVelocities", np.max(np.abs(velocities)))
        # # print "Mean vel: " + str(np.mean(np.mean(np.abs(velocities), 1),1))
        # # print "Max vel: " + str(np.max(np.max(np.abs(velocities), 1),1))
        #
        # # goal_generator_diagnostics = self._goal_generator.get_diagnostics()
        # # for key, val in goal_generator_diagnostics.items():
        # #   logger.record_tabular(key, val)
        #
        # # action_limiter_diagnostics = self._action_limiter.get_diagnostics()
        # # for key, val in action_limiter_diagnostics.items():
        # #    logger.record_tabular(key, val)
        #
        # # self.update_failure_rate(paths)
        #
        # # action_limit = self._action_limiter.get_action_limit()
        # # failure_rate = self.get_mean_failure_rate()
        # # expected_damage = action_limit * failure_rate
        # # logger.record_tabular('Expected Damage', expected_damage)


