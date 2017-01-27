import numpy as np
import pickle
import gym
# import fetch_setup
import sandbox.dave.calibration.calibration_config as calibration_config
# from conopt.worldgen.world import World

import time
from sandbox.dave.pr2.action_limiter import FixedActionLimiter

from sandbox.dave.rllab.envs.mujoco.pr2_env_calibration import Pr2EnvLego
from rllab.envs.normalized_env import normalize

action_limiter = FixedActionLimiter()


import xml.dom.minidom as dom

default_actuatorgains = [1 \
                         for d in dom.parseString(calibration_config.default_xml).\
                         getElementsByTagName('actuator')[0].getElementsByTagName('motor')]

# TODO: Refactor as class?

def set_joint_params(xml, joint_param_dict, scale_params=True):
    xml = dom.parseString(xml)
    joints = xml.getElementsByTagName('worldbody')[0].getElementsByTagName('joint')
    actuators = xml.getElementsByTagName('actuator')[0].getElementsByTagName('motor')
    inertials = xml.getElementsByTagName('inertial')
    for joint, joint_data in joint_param_dict.items():
        for param, param_val in joint_data.items():
            if scale_params:
                param_val = calibration_config.unscale_param(param, param_val)
            param_type, param_name = param.split('_')
            if param_type == 'joint':
                joint_idx = calibration_config.joint_indices[joint]
                joints[joint_idx].setAttribute(param_name, str(param_val))
            elif param_type == 'actuator':
                actuator_idx = calibration_config.actuator_indices[joint]
                actuators[actuator_idx].setAttribute(param_name,
                                                     str(param_val * default_actuatorgains[actuator_idx]))

            elif param_type == 'inertial':
                inertial_idx = calibration_config.inertial_indices[joint]
                inertials[inertial_idx].setAttribute(param_name, 
                                                     str(param_val))
            else:
                assert(False)

    new_xml = xml.toxml()
    return new_xml

def get_default(joint, param):
    param_type, param_name = param.split('_')
    xml = dom.parseString(calibration_config.default_xml)
    result = None
    if param_type == 'joint':
        joint_idx = calibration_config.joint_indices[joint]
        joint = xml.getElementsByTagName('worldbody')[0].getElementsByTagName('joint')[joint_idx]
        classname = None
        parent = joint.parentNode
        print(joint, parent)
        while True:
            if parent is None:
                break
            if parent.hasAttribute('childclass'):
                classname = parent.getAttribute('childclass')
                break
            parent = parent.parentNode
        if classname is not None:
            defaults = xml.getElementsByTagName('default')
            classdom = defaults[[d.getAttribute('class') for d in defaults].index(classname)]
            while classdom.nodeName == 'default':
                joint = classdom.getElementsByTagName('joint')
                if joint and joint[0].hasAttribute(param_name):
                    result = float(joint[0].getAttribute(param_name))
                    break
                classdom = classdom.parentNode
    return result

def get_joint_params(xml, params, scale_params=True):
    xml = dom.parseString(xml)
    joints = xml.getElementsByTagName('worldbody')[0].getElementsByTagName('joint')
    actuators = xml.getElementsByTagName('actuator')[0].getElementsByTagName('motor')
    inertials = xml.getElementsByTagName('inertial')
    import pdb; pdb.set_trace()
    results = {}
    for joint, joint_params in params.items():
        results[joint] = {}
        for param in joint_params:
            param_type, param_name = param.split('_')
            if param_type == 'joint':
                joint_idx = calibration_config.joint_indices[joint]
                if joints[joint_idx].hasAttribute(param_name):
                    unscaled = float(joints[joint_idx].getAttribute(param_name))
                else:
                    unscaled = get_default(joint, param)
            elif param_type == 'inertial':
                inertial_idx = calibration_config.inertial_indices[joint]
                if inertials[inertial_idx].hasAttribute(param_name):
                    unscaled = float(inertials[inertial_idx].getAttribute(param_name))
                else:
                    unscaled = None 
            elif param_type == 'actuator':
                actuator_idx = calibration_config.actuator_indices[joint]
                if actuators[actuator_idx].hasAttribute(param_name):
                    unscaled = float(actuators[actuator_idx].getAttribute(param_name)) / default_actuatorgains[actuator_idx]
                else:
                    assert(False)
            else:
                assert(False)
            if scale_params:
                results[joint][param] = calibration_config.scale_param(param, unscaled)
            else:
                results[joint][param] = unscaled

    return results

    

def build_env(param_updates={}, env_type='SimFetchShort-v0', env=None,
              scale_params=True):
    env = normalize(Pr2EnvLego(
        action_limiter=action_limiter,
        max_action=1,
        pos_normal_sample=False,
        qvel_init_std=0.01,
        # use_vision=True,
    ))
    import copy
    damping = copy.copy(env._wrapped_env.model.dof_damping)
    stiffness = copy.copy(env._wrapped_env.model.jnt_stiffness)
    armature = copy.copy(env._wrapped_env.model.dof_armature)
    frictionloss = copy.copy(env._wrapped_env.model.dof_frictionloss)
    joint_idx = 0
    damping[joint_idx] = 0.0025000000000000001 * 200
    frictionloss[joint_idx] = 0
    armature[joint_idx] = 0.5 * 200
    #
    joint_idx = 1
    damping[joint_idx] = 0.021533690279516252 * 200
    frictionloss[joint_idx] = 0.66265139932791739
    armature[joint_idx] = 0.0061718509830245346 * 200
    #
    joint_idx = 2
    damping[joint_idx] =  0.00023919592681921829 * 200
    frictionloss[joint_idx] = 0.076386059377093307
    armature[joint_idx] = 0.0012230499299831013 * 200
    #
    joint_idx = 3
    damping[joint_idx] = 0.01008434976111381 * 200
    frictionloss[joint_idx] = 0
    armature[joint_idx] = 0.0012785375495622342 * 200
    #
    joint_idx = 4
    damping[joint_idx] = 0.0020215130664460843 * 200
    frictionloss[joint_idx] = 0.00061591843955659286
    armature[joint_idx] = 0.00065406342495096946 * 200
    #
    joint_idx = 5
    damping[joint_idx] = 0.012856592873849996 * 200
    frictionloss[joint_idx] = 0
    armature[joint_idx] = 0.0015341928736678926 * 200
    #
    joint_idx = 6
    damping[joint_idx] = 0.0024824916289231075 * 200
    frictionloss[joint_idx] = 0
    armature[joint_idx] = 0.00085692998114568215 * 200

    env._wrapped_env.model.jnt_stiffness = stiffness
    env._wrapped_env.model.dof_damping = damping
    env._wrapped_env.model.dof_armature = armature
    return env
