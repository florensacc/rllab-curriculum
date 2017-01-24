#!/usr/bin/env python
# from fetch_setup.logger import get_logger
import sys
import os
import click
import six.moves.cPickle as pickle
import gym
# import fetch_setup
import numpy as np
import calibration_config as calibration_config

import roslib
roslib.load_manifest('simple_torque_controller')
import rospy
from std_msgs.msg import Float64MultiArray

import numpy as np
from subprocess import Popen, check_output
import time

class PR2TorquesEnv(object):
    def __init__(self):
        rospy.init_node('rllab_torque_publisher', anonymous=True)

        # kill old controller, load & start new controller
        rospy.set_param('simple_torque_controller/SimpleTorqueController/type',
            'simple_torque_controller/SimpleTorqueController')
        output = check_output("rosrun pr2_controller_manager pr2_controller_manager list", shell=True)
        output_lines = output.split('\n')
        to_kill_l_arm = False
        to_load_controller = True
        to_start_controller = True
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

    def get_joint_angles(self):
        return self.obs_state[0:7]

    def get_joint_velocities(self):
        return self.obs_state[7:14]

    def get_end_effector_position(self):
        return self.obs_state[14:17]

    def apply_torques(self, joint_torques):
        current_msg = Float64MultiArray(data=joint_torques)
        self.torque_array_pub.publish(current_msg)  
        # Wait 50 milliseconds before applying a new torque.
        time.sleep(0.05)

    def apply_torques_position(self, joint_torques):
        current_msg = Float64MultiArray(data=joint_torques)
        self.torque_array_pub.publish(current_msg)  
        # Wait 50 milliseconds before applying a new torque.
        time.sleep(0.01)

   
def linear_chirp(N=100, chirpfreq=10, chirprate=10, ampfreq=2):
    """ Defines a smooth trajectory for the robot's limbs """
    X = np.linspace(0, 1, N)
    amp = np.cos(ampfreq * 2 * np.pi * X)
    freq = chirpfreq * X + 0.5 * chirprate * (X ** 2)
    chirp = np.sin(freq * 2 * np.pi)
    return amp * chirp

@click.command()
@click.option('--path', type=str, default="./calibration", 
              help='directory where to write results')
@click.option('--fast', is_flag=True, default=False, help='Gather trajectories at 100hz')
@click.option('--test_data', is_flag=False, default=False, help='Gather test trajectories')
def main(path, fast, test_data):
    if not os.path.exists(path):
        os.makedirs(path)

    env = PR2TorquesEnv()
    kp = np.array([5, 4, 10, 10, 1, 8, 8])
    kd = np.array(7 * [1])

    joints = ["l_shoulder_pan_joint", \
              "l_shoulder_lift_joint", \
              "l_upper_arm_roll_joint", \
              "l_elbow_flex_joint", \
              "l_forearm_roll_joint", \
              "l_wrist_flex_joint", \
              "l_wrist_roll_joint", \
             ]
    scale = 0.5
    if not test_data:
        chirp_frequencies = scale * np.array([1., 0., -1.])
        chirp_rates = scale * np.array([-1., 1., 2.])
        amp_frequencies = scale * np.array([-1., 0., 1.])
    else:
        chirp_frequencies = scale * np.array([-0.75, 0.75])
        chirp_rates = scale * np.array([-0.75, 0.0, 0.75])
        amp_frequencies = scale * np.array([-0.75, 0.75])
    fidx = 0
    traj_len = 100
    # Joint-specific trajectories
    
    for idx, joint in enumerate(joints):
        print("Processing joint: %s", joint)
        for chirp_freq in chirp_frequencies:
            for chirp_rate in chirp_rates:
                for amp_freq in amp_frequencies:
                    print("Chirp frequency = %f, chirp rate = %f, " \
                                + "amp frequency = %f", chirp_freq, chirp_rate, amp_freq)
                    qpos, qvel, ctrl = [], [], []
                    qpos.append(env.get_joint_angles())
                    qvel.append(env.get_joint_velocities())
                    ctrl_scale = 3 #np.min(np.abs(calibration_config.joint_ranges[joint]))
                    controls = linear_chirp(traj_len, chirp_freq, chirp_rate, amp_freq) * ctrl_scale
                    for i in range(traj_len):
                    	torque = np.zeros((7,))
                        # for _ in range():
                        torque = kp * (-env.obs_state[:7]) + kd * (-env.obs_state[7:14])
                            # torque[idx] = 0
                            # env.apply_torques_position(torque)
                        # print(env.obs_state[:14])
                    	# torque[idx] = controls[i]
                        # print(controls[i])
                       	env.apply_torques(torque)
                    	qpos.append(env.get_joint_angles())
                    	qvel.append(env.get_joint_velocities())
                    	ctrl.append(torque)

                    data = {"joints": [joints[idx]], \
                        "ctrl": ctrl, \
                        "qpos": qpos, \
                        "qvel": qvel}
                    if not fast and not test_data:
                        out_file = "%s/calibration_%07d.pkl"%(path, fidx) 
                    elif not fast and test_data:
                        out_file = "%s/test_%07d.pkl"%(path, fidx)
                    elif fast:
                        out_file = "%s/fast_%07d.pkl"%(path, fidx) 
                    with open(out_file, "wb") as f:
                        pickle.dump(data, f)
                    fidx += 1
    

    # Whole-arm trajectories
    if not test_data: 
        np.random.seed(0)
        num_whole_arm = 30
    else:
        np.random.seed(142857)
        num_whole_arm = 20
    for idx in range(num_whole_arm):
        print("Whole arm trajectory %d/%d"%(idx, num_whole_arm))
        controls = [linear_chirp(traj_len, *np.random.uniform(-1, 1, 3)) for _ in range(7)]
        for _ in range(2):
            controls += [0.05 * np.ones(traj_len)]
        controls = np.stack(controls).T
        qpos, qvel, ctrl = [], [], []
        qpos.append(env.get_joint_angles())
        qvel.append(env.get_joint_velocities())


        for i in range(traj_len):
	    	torque = controls[i][:7] * ctrl_scale
           	env.apply_torques(torque)
        	qpos.append(env.get_joint_angles())
        	qvel.append(env.get_joint_velocities())
        	ctrl.append(controls[i, :])

        data = {"joints": joints, \
                "ctrl": ctrl, \
                "qpos": qpos, \
                "qvel": qvel, 
               }
        if not fast and not test_data:
            out_file = "%s/calibration_%07d.pkl"%(path, fidx + idx) 
        elif not fast and test_data:
            out_file = "%s/test_%07d.pkl"%(path, fidx + idx)
        elif fast:
            out_file = "%s/fast_%07d.pkl"%(path, fidx + idx) 
        with open(out_file, "wb") as f:
            pickle.dump(data, f)

    exit(0)

if __name__ == '__main__':
    main()