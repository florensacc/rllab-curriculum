from __future__ import print_function

import roslib
roslib.load_manifest('simple_torque_controller')
import rospy
from std_msgs.msg import Float64MultiArray

import numpy as np
from subprocess import Popen, check_output

from sandbox.dave.rllab.envs.mujoco.mujoco_env import MujocoEnv
from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

import time


class Pr2EnvReal(MujocoEnv, Serializable):

    FILE = 'pr2_legofree.xml'

    def __init__(
            self,
            goal_generator=None,
            action_penalty_weight=0.001,
            distance_thresh=0.01,  # 1 cm
            model='pr2_legofree.xml',
            max_action=float("inf"),
            action_limiter=FixedActionLimiter(),
            use_running_average_failure_rate=True,
            failure_rate_gamma=0.9,
            mean_failure_rate_init=1.0,
            offset=np.zeros(3),
            *args, **kwargs):

        self.action_penalty_weight = action_penalty_weight
        self.distance_thresh = distance_thresh
        self._goal_generator = goal_generator
        self._action_limiter = action_limiter
        self.goal = None
        if model not in [None, 0]:
            self.set_model(model)
        self.action_limit = max_action
        self.max_action_limit = 3
        self.min_action_limit = 0.1
        self.goal_dims = 3
        self.mean_failure_rate = mean_failure_rate_init
        self.failure_rate_gamma = failure_rate_gamma
        self.use_running_average_failure_rate = use_running_average_failure_rate
        self.offset = offset
        self.distance_tip_lego_penalty_weight = 0.6
        self.angle_penalty_weight = 0.2
        self.occlusion_weight = 0.0005

        self.init_pr2_real()

        super(Pr2EnvReal, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    # ---- Methods that interface with the PR2 ------

    def init_pr2_real(self):
        rospy.init_node('rllab_torque_publisher', anonymous=True)

        # kill old controller, load & start new controller
        rospy.set_param('simple_torque_controller/SimpleTorqueController/type',
                        'simple_torque_controller/SimpleTorqueController')
        output = check_output("rosrun pr2_controller_manager pr2_controller_manager list", shell=True)
        output_lines = output.split('\n')
        to_kill_l_arm = False
        to_load_controller = False
        to_start_controller = False
        for line in output_lines:
            if 'l_arm_controller' in line:
                print(line)
                to_kill_l_arm = True
            if 'simple_torque_controller' in line:
                print(line)
                to_load_controller = False
                if 'running' in line:
                    to_start_controller = False
        if to_kill_l_arm:
            kill_l_arm = Popen("rosrun pr2_controller_manager pr2_controller_manager " \
                               + "kill l_arm_controller", shell=True)
            rospy.sleep(0.2)
        if to_load_controller:
            load_simple_torque = Popen("rosrun pr2_controller_manager pr2_controller_manager " \
                                       + "load simple_torque_controller/SimpleTorqueController", shell=True)
            rospy.sleep(0.2)
        if to_start_controller:
            start_simple_torque = Popen("rosrun pr2_controller_manager pr2_controller_manager " \
                                        + "start simple_torque_controller/SimpleTorqueController", shell=True)
            rospy.sleep(0.2)

        self.torque_array_pub = rospy.Publisher('/rllab_torque', Float64MultiArray)
        self.obs_state = np.zeros((17,))

        rospy.Subscriber("/rllab_obs", Float64MultiArray, self.obs_callback)

    def obs_callback(self, msg):
        self.obs_state = np.array(msg.data)

    # Returns joint angles as a numpy array:
    # [l_shoulder_pan_joint, l_shoulder_lift_joint, l_upper_arm_roll_joint, l_elbow_flex_joint, l_forearm_roll_joint, l_wrist_flex_joint, l_wrist_roll_joint]
    def get_positions(self):
        dim = self.model.data.qpos.shape[0]
        joint_angles = self.obs_state[0:7]
        lego_position = np.array([0.6, 0.2, 0.5025, 0, 0, 0, 1])
        #return self.model.data.qpos.flat[:-3]
        #self.set_mujoco_pr2_pos(joint_angles)
        return np.concatenate([joint_angles, lego_position])

    # Returns joint angles as a numpy array (same order as get_joint_angles)
    def get_velocities(self):
        dim = self.model.data.qpos.shape[0]
        joint_vels = self.obs_state[7:14]
        lego_velocity = np.array([0, 0, 0])
        #return self.model.data.qvel.flat[:-3]
        #self.set_mujoco_pr2_vel(joint_vels)
        return np.concatenate([joint_vels, lego_velocity])

    # Returns the x,y,z coordinates of the end-effector (l_gripper_tool_frame).
    def get_tip_position(self):
        ee_pos = self.obs_state[14:17]
        #return self.model.data.site_xpos[0]
        return ee_pos

    # Input: joint torques as a numpy array (same order as get_joint_angles)
    # Effect: Applies joint torques to the PR2
    def apply_torques(self, joint_torques):
        # Apply torques to the simulated PR2 model.
        #self.forward_dynamics(joint_torques)
        # Apply torques to the real PR2.
        #print(joint_torques)
        current_msg = Float64MultiArray(data=joint_torques)
        self.torque_array_pub.publish(current_msg)
        # Wait 50 milliseconds before applying a new torque.
        time.sleep(0.05)

    # ---- Set Mujoco environment to match real PR2-----

    # def set_mujoco_pr2_pos(self, joint_angles):
    #     qpos = np.hstack((joint_angles, self.goal))
    #     self.model.data.qpos = qpos
    #
    # def set_mujoco_pr2_vel(self, joint_vels):
    #     qvel = np.concatenate([joint_vels, np.zeros(self.goal_dims)])
    #     self.model.data.qvel = qvel

    # -----------------------------------------------

    def get_lego_position(self):
        #Return the position of the lego block
        return self.model.data.site_xpos[-1]

    def get_vec_to_goal(self):
        tip_position = self.get_tip_position()
        # Compute the distance to the goal
        vec_to_goal = tip_position - (self.goal + self.offset)
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

    def get_current_obs(self):
        dim = self.model.data.qpos.shape[0]

        return np.concatenate([
            self.get_positions(),
            # self.model.data.qpos.flat[:-3], # We do not need to explicitly include the goal
            #                                 # since we already have the vec to the goal.
            self.get_velocities(),  # Do not include the velocity of the target (should be 0).
            self.get_tip_position(),
            self.get_vec_tip_to_lego(),
            self.get_vec_to_goal(),
        ]).reshape(-1)

    def step(self, action):

        # Limit actions to the specified range.
        action_limit = self.action_limit * self.action_space.ones()
        #action_limit = self.action_limit * np.ones(self.action_space.shape)
        action = np.maximum(action, -action_limit)
        action = np.minimum(action, action_limit)
        vec_tip_to_lego = self.get_vec_tip_to_lego()

        # Simulate this action and get the resulting state.
        self.apply_torques(action)

        vec_to_goal = self.get_vec_to_goal()
        distance_to_goal = np.linalg.norm(vec_to_goal)
        distance_tip_to_lego = np.linalg.norm(vec_tip_to_lego)

        # Penalize the robot for being far from the goal and for having the arm far from the lego.
        reward_dist = - distance_to_goal
        reward_tip = - self.distance_tip_lego_penalty_weight * distance_tip_to_lego

        cos_angle = self.get_cos_vecs()
        reward_angle = - self.angle_penalty_weight * cos_angle

        # Penalize the robot for large actions.f
        # reward_occlusion = self.occlusion_weight * self.get_reward_occlusion()
        reward_ctrl = - self.action_penalty_weight * np.square(action).sum()
        reward = reward_dist + reward_tip + reward_ctrl + reward_angle #+ reward_occlusion
        state = self._state
        # print(reward_occlusion, reward_angle, reward_tip, reward_dist, )
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self.get_current_obs()

        return Step(ob, float(reward), done,  # not self.do_rand,
                    distance_to_goal=distance_to_goal,
                    distance_to_goal_x=vec_to_goal[0],
                    distance_to_goal_y=vec_to_goal[1],
                    distance_to_goal_z=vec_to_goal[2],
                    distance_tip_to_lego=distance_tip_to_lego,
                    reward_dist=reward_dist,
                    reward_tip=reward_tip,
                    reward_angle=reward_angle,
                    # reward_occlusion=reward_occlusion,
                    )

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.5

    @overrides
    def reset_mujoco(self):
        goal_dims = 3  # self.goal_dims
        lego_dims = 6
        # Use current position as new position.
        qpos_curr = self.model.data.qpos #[:-goal_dims]
        qpos = list(qpos_curr)

        # Generate a new goal.
        lego_position = self.get_lego_position()
        if self._goal_generator is not None:
            self.goal = self._goal_generator.generate_goal(lego_position)
            qpos[-self.goal_dims:] = self.goal[:, None]
        else:
            print("No goal generator!")

        # Use the current velocity as the new velocity.
        qvel = np.array(self.model.data.qvel)

        # Set the velocity of the goal (the goal should be stationary).
        qvel[-self.goal_dims:] = 0

        # Set the goal position and velocity.
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

        if self._action_limiter is not None:
            self.action_limit = self._action_limiter.get_action_limit()

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

        distances_to_goal_x = [path["env_infos"]["distance_to_goal_x"] for path in paths]
        distances_to_goal_y = [path["env_infos"]["distance_to_goal_y"] for path in paths]
        distances_to_goal_z = [path["env_infos"]["distance_to_goal_z"] for path in paths]

        logger.record_tabular('FinalDistanceToGoalX', np.mean([d[-1] for d in distances_to_goal_x]))
        logger.record_tabular('FinalDistanceToGoalY', np.mean([d[-1] for d in distances_to_goal_y]))
        logger.record_tabular('FinalDistanceToGoalZ', np.mean([d[-1] for d in distances_to_goal_z]))

        logger.record_tabular('MaxFinalDistanceToGoalX', np.max([d[-1] for d in distances_to_goal_x]))
        logger.record_tabular('MaxFinalDistanceToGoalY', np.max([d[-1] for d in distances_to_goal_y]))
        logger.record_tabular('MaxFinalDistanceToGoalZ', np.max([d[-1] for d in distances_to_goal_z]))

        logger.record_tabular('MinFinalDistanceToGoalX', np.min([d[-1] for d in distances_to_goal_x]))
        logger.record_tabular('MinFinalDistanceToGoalY', np.min([d[-1] for d in distances_to_goal_y]))
        logger.record_tabular('MinFinalDistanceToGoalZ', np.min([d[-1] for d in distances_to_goal_z]))

        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        logger.record_tabular('MinDistanceToGoal', np.mean([np.min(d) for d in distances_to_goal]))
        logger.record_tabular('FinalDistanceToGoal', np.mean([d[-1] for d in distances_to_goal]))

        # The task is considered complete when we get within distance_thresh of the goal.
        # reached_goal_indices = np.where(distances_to_goal < distance_thresh)
        # if (distances_to_goal < distance_thresh).any():

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

        goal_generator_diagnostics = self._goal_generator.get_diagnostics()
        for key, val in goal_generator_diagnostics.items():
            logger.record_tabular(key, val)

        action_limiter_diagnostics = self._action_limiter.get_diagnostics()
        for key, val in action_limiter_diagnostics.items():
            logger.record_tabular(key, val)

        self.update_failure_rate(paths)

        action_limit = self._action_limiter.get_action_limit()
        failure_rate = self.get_mean_failure_rate()
        expected_damage = action_limit * failure_rate
        logger.record_tabular('Expected Damage', expected_damage)

    def set_model(self, model):
        self.__class__.FILE = model
