import six
# if not six.PY2:
    # import conopt
import os
import xml.dom.minidom as dom
import numpy as np



joints =  ["l_shoulder_pan_joint", \
              "l_shoulder_lift_joint", \
              "l_upper_arm_roll_joint", \
              "l_elbow_flex_joint", \
              "l_forearm_roll_joint", \
              "l_wrist_flex_joint", \
              "l_wrist_roll_joint", \
             ]

joint_indices = dict([(joint, i) for i, joint in enumerate(joints)])

actuators = joints[3:]
actuator_indices = dict([(actuator, i) for i, actuator in enumerate(actuators)])

params = [
    'joint_damping',
    'joint_frictionloss',
    'joint_armature',
    # 'joint_stiffness',
    'inertial_mass',
    'inertial_diaginertia'
]
param_ranges = {
    # 'actuator_kp': (0.1, 10),
    'joint_damping': (0, 200),
    'joint_frictionloss': (0, 1),
    'joint_armature': (0, 200),
    # 'inertial_mass': (0., 100.),
}


ctrl_ranges = { "l_shoulder_pan_joint": (-3, 3),
              "l_shoulder_lift_joint": (-3, 3),
              "l_upper_arm_roll_joint": (-3, 3),
              "l_elbow_flex_joint": (-3, 3),
              "l_forearm_roll_joint": (-3, 3),
              "l_wrist_flex_joint": (-3, 3),
              "l_wrist_roll_joint": (-3, 3),
}

joint_ranges = { "l_shoulder_pan_joint": (-0.714602, 2.285398),
              "l_shoulder_lift_joint": (-0.5236, 1.3963),
              "l_upper_arm_roll_joint": (-0.8, 3.9),
              "l_elbow_flex_joint": (-2.3213, 0),
              "l_forearm_roll_joint": (-3.14, 3.14),
              "l_wrist_flex_joint": (-2.094, 0),
              "l_wrist_roll_joint": (-3.14, 3.14),
             }

if not six.PY2:
    default_xml_path = '/home/young_clgan/GitRepos/rllab-private/sandbox/dave/vendor/mujoco_models/pr2_lego_calibration.xml'
    default_xml = dom.parse(default_xml_path).toxml()
else:
    default_xml_path = '/home/young_clgan/GitRepos/rllab-private/sandbox/dave/vendor/mujoco_models/pr2_lego_calibration.xml'
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

############################# RESULTS ##############################
#  {'l_wrist_roll_joint': {'joint_armature': 0.00085692998114568215, 'joint_stiffness': 2.636213591410778e-12, 'joint_damping': 0.0024824916289231075, 'joint_frictionloss': 2.4518251058590471e-38}}
#
#
#         {'l_forearm_roll_joint': {'joint_armature': 0.00065406342495096946, 'joint_stiffness': 5.4327634934188412e-06,
#                                   'joint_damping': 0.0020215130664460843, 'joint_frictionloss': 0.00061591843955659286}}
#
#
#{'l_elbow_flex_joint': {'joint_armature': 0.0012785375495622342, 'joint_stiffness': 0.028478671428145495, 'joint_damping': 0.01008434976111381, 'joint_frictionloss': 2.4038869746193596e-07}}
#
#
# {'l_upper_arm_roll_joint': {'joint_armature': 0.0012230499299831013, 'joint_stiffness': 0.06654510055176617, 'joint_damping': 0.00023919592681921829, 'joint_frictionloss': 0.076386059377093307}}
#
#
#{'l_shoulder_pan_joint': {'joint_damping': 0.0025000000000000001, 'joint_frictionloss': 1.98392171387699e-15, 'joint_armature': 0.5, 'joint_stiffness': 0.14999999999999999}}
#
#
# {'l_wrist_flex_joint': {'joint_damping': 0.012856592873849996, 'joint_frictionloss': 3.3673079988621432e-07, 'joint_armature': 0.0015341928736678926, 'joint_stiffness': 0.040430440641066732}}
#
#
#{'l_shoulder_lift_joint': {'joint_damping': 0.021533690279516252, 'joint_frictionloss': 0.66265139932791739, 'joint_armature': 0.0061718509830245346, 'joint_stiffness': 0.3466144216446384}}
#