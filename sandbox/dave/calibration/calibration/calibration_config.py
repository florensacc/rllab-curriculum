import six
if not six.PY2:
    import conopt
import os
import xml.dom.minidom as dom
import numpy as np



joints = [
        'torso_lift_joint',
        'head_pan_joint',
        'head_tilt_joint',
        'shoulder_pan_joint',
        'shoulder_lift_joint', 
        'upperarm_roll_joint',  
        'elbow_flex_joint', 
        'forearm_roll_joint',  
        'wrist_flex_joint', 
        'wrist_roll_joint', 
        'l_gripper_finger_joint', 
        'r_gripper_finger_joint', 
]

joint_indices = dict([(joint, i) for i, joint in enumerate(joints)])

actuators = joints[3:]
actuator_indices = dict([(actuator, i) for i, actuator in enumerate(actuators)])

params = [
    'actuator_kp',
    'joint_damping',
    'joint_frictionloss',
    'joint_armature',
    'joint_stiffness',
    'inertial_mass',
    'inertial_diaginertia'
]
param_ranges = {
    'actuator_kp': (0.1, 10),
    'joint_damping': (0, 100),
    'joint_frictionloss': (0, 0.05),
    'joint_armature': (0.1, 10),
    'joint_stiffness': (0, 10),
    'inertial_mass': (0., 100.),
}

joint_ranges = {
    'shoulder_pan_joint': (-1.6056, 1.6056),
    'shoulder_lift_joint': (-1.221, 1.518),
    'upperarm_roll_joint': (-3.14, 3.14),
    'elbow_flex_joint': (-2.251, 2.251),
    'forearm_roll_joint': (-3.14, 3.14),
    'wrist_flex_joint': (-2.16, 2.16),
    'wrist_roll_joint': (-3.14, 3.14),
    'l_gripper_finger_joint': (0.0, 0.05),
}
if not six.PY2:
    default_xml_path = os.path('/home/iclavera/pr2/rllab-private/sandbox/dave/vendor/mujoco_models/pr2_legofree.xml')
    default_xml = dom.parse(default_xml_path).toxml()
else:
    default_xml_path = '/home/iclavera/pr2/rllab-private/sandbox/dave/vendor/mujoco_models/pr2_legofree.xml'
    default_xml = dom.parse(default_xml_path).toxml()

def scale_param(param, val):
    try:
        param_type = param.split('_')[0]
        if param_type == 'joint' or param_type == 'inertial':
            result = (val - param_ranges[param][0]) / (param_ranges[param][1] - 
                                                       param_ranges[param][0])
        elif param_type == 'actuator':
            result = (np.log(val) - np.log(param_ranges[param][0])) / \
                     (np.log(param_ranges[param][1]) - np.log(param_ranges[param][0]))
        else:
            raise NotImplementedError("No rule for param type %s"%param_type)
        return result
    
    except KeyError:
        raise ValueError("No range for parameter %s provided"%param)

def unscale_param(param, val):
    try:
        param_type = param.split('_')[0]
        if param_type == 'joint' or param_type == 'inertial':
            result = val * (param_ranges[param][1] - param_ranges[param][0]) + param_ranges[param][0]
        elif param_type == 'actuator':
            result = param_ranges[param][0] * np.e ** (
                    val * (np.log(param_ranges[param][1]) - np.log(param_ranges[param][0])))
        else:
            raise NotImplementedError("No rule for param type %s"%param_type)

        return result
    
    except KeyError:
        raise ValueError("No range for parameter %s provided"%param)
