ka#!/usr/bin/env python
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

def main():
    pr2_env = MockRLLabPR2Env()
    torque_ind = 0
    A = 5  
    w = 110 * np.pi
    t = 0
    dt = 0.01
    time_steps = 200
    file = open("calibration_data.csv", 'wb')
    last_time = time.clock()
    writer = csv.writer(file, delimiter=',')
    number = map(str, range(7))
    header = ['torque' + s for s in number] + ['angle_after,' + s for s in number]  \
    + ['angle_after' + s for s in number] + ['velocity_before' + s for s in number] \
    + ['velocity_after' + s for s in number]
    writer.writerow(header)
    while not rospy.is_shutdown():
        t +=1
        time.sleep(0.1)
        cur_time = time.clock()
        new_torques = np.zeros((7,))

        # dt = (cur_time - last_time)
        
        A += np.random.normal(0, dt)
        w += np.random.normal(0, dt) 

        print('dt', dt, '\n A', A, 'w', w,'\n\n')

        new_torques[torque_ind] = A * np.sin(w * cur_time)
        last_time = cur_time

        angles_before = pr2_env.get_joint_angles()
        velocities_before = pr2_env.get_joint_velocities()

        pr2_env.apply_torques(new_torques)

        angles_after = pr2_env.get_joint_angles()
        velocities_after = pr2_env.get_joint_velocities()
        to_write = np.concatenate([new_torques, angles_before, angles_after, \
         velocities_before, velocities_after])
        writer.writerow(to_write)
        if t % time_steps == 0:
            torque_ind += 1
            if torque_ind >= 7:
                print('CALIBRATION FINISHED')
                A = 5.5
                new_torques = np.zeros((7,))
                pr2_env.apply_torques(new_torques)
                rospy.spin()

if __name__ == '__main__':
    main()