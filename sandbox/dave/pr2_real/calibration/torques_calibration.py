#!/usr/bin/env python
import roslib


roslib.load_manifest('simple_torque_controller')
import rospy
from std_msgs.msg import Float64MultiArray

import numpy as np
from subprocess import Popen, check_output
import time
import csv

class MockRLLabPR2Env(object):
    def __init__(self):
        rospy.init_node('rllab_torque_publisher')

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
                print
                line
                to_kill_l_arm = True
            if 'simple_torque_controller' in line:
                print
                line
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


def main():
    pr2_env = MockRLLabPR2Env()
    torque_ind = 0
    A = 0
    file = open("calibration_data.csv", 'wb')
    last_time = time.clock()
    writer = csv.writer(file, delimiter=',')
    number = map(str, range(7))
    header = ['torque' + s for s in number] + ['position' + s for s in number] + ['velocity' + s for s in number]
    writer.writerow(header)
    while not rospy.is_shutdown():
        cur_time = time.clock()
        new_torques = np.zeros((7,))

        A += np.random.uniform() * (cur_time - last_time)
        new_torques[torque_ind] = A * np.sin(12 * cur_time)
        last_time = cur_time

        pr2_env.apply_torques(new_torques)
        rospy.sleep(0.05)
        angles = pr2_env.get_joint_angles()
        velocities = pr2_env.get_joint_velocities()
        writer.writerows(np.concatenate[new_torques, angles, velocities])


if __name__ == '__main__':
    main()