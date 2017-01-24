import numpy as np
import pickle
import copy
import os
import gym
# import fetch_setup
import functools
import scipy.optimize
import sys
import math
# from fetch_setup.logger import get_logger
import sandbox.dave.calibration.calibration_config as calibration_config
from sandbox.dave.calibration.xml_utils import *
from sandbox.dave.pr2.action_limiter import FixedActionLimiter

from sandbox.dave.rllab.envs.mujoco.pr2_env_calibration import Pr2EnvLego
from rllab.envs.normalized_env import normalize

action_limiter = FixedActionLimiter()


# logger = get_logger("calibration_utils")


def rollouts(env, actions, start_qpos=None):
    observations = []
    # actions = [actions[5], actions[9]]
    for i, traj in enumerate(actions):
        traj_obs = []
        if start_qpos is not None:
            env._wrapped_env.reset_mujoco(qpos=start_qpos[i], qvel=np.zeros_like(start_qpos[i]))
            obs = start_qpos[i]
        else:
            obs = env.wrapped_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        traj_obs.append(obs)
        for a in traj:
            if len(a) > 7:
                a = a[:7]
            obs = env.step(a)
            if isinstance(obs, tuple):
                obs = obs[0]
                obs = obs[:7]
                # qvel = obs[7:]
                # obs = np.concatenate([qpos, qvel])
            traj_obs.append(obs)

        traj_obs = np.stack(traj_obs)
        observations.append(traj_obs)
    return np.stack(observations)

def load_calibration_data(joints, data_path):
    """
    returns:
        train_data: a dict of the form {joint: {'qpos': [np.array([[...]]), ...],
                                          'ctrl': [np.array([[...]]), ...]}}
        test_data: a dict of the same form with test trajectories
    """
    train_data = {}
    test_data = {}
    for joint in joints:
        train_data[joint] = {'qpos': [], 'ctrl': []}
        test_data[joint] = {'qpos': [], 'ctrl': []}
    
    for file in [f for f in os.listdir(data_path) if f.endswith(".pkl")]:
        with open(os.path.join(data_path, file), 'rb') as f:
            if sys.version_info[0] == 2:
                data = pickle.load(f)
            else:
                data = pickle.load(f, encoding='latin1')
        for joint in set(joints).intersection(set(data['joints'])):
            if file.startswith("test"):
                test_data[joint]['qpos'].append(np.stack(data['qpos']))
                test_data[joint]['ctrl'].append(np.stack(data['ctrl']))
            elif file.startswith("calibration"):
                train_data[joint]['qpos'].append(np.stack(data['qpos']))
                train_data[joint]['ctrl'].append(np.stack(data['ctrl']))

    return train_data, test_data

def evaluate_rollout(real_qpos, sim_qpos, joint=None):
    if joint is None:
        active_joints = calibration_config.joints
    else:
        active_joints = [joint]
    joint_ranges = dict([(k, (v[1] - v[0])/2) for k, v in calibration_config.joint_ranges.items()])
    err = 0.
    for j in active_joints:
        joint_idx = calibration_config.joint_indices[j]
        err += np.mean(np.abs(real_qpos[:, joint_idx] - sim_qpos[:, joint_idx])) / joint_ranges[j]
    return err / len(active_joints)

def evaluate_xml(test_data, param_updates={}):
    env = build_env(param_updates=param_updates)
    results = {}
    for joint in test_data.keys():
        actions = test_data[joint]['ctrl']
        fake_qpos = test_data[joint]['qpos']
        start_pos = [qp[0] for qp in fake_qpos]
        real_qpos = rollouts(env, actions, start_pos)
        results[joint] = np.mean([evaluate_rollout(rqp, sqp, joint) 
                                 for rqp, sqp in zip(real_qpos, fake_qpos)])
    print('Evaluated Results')
    print(results)
    return results

def calibrate_xml(train_data, params):
    result = {}
    for joint in train_data.keys():
        joint_idx = calibration_config.joint_indices[joint]
        print("Calibrating joint %s", joint)
        joint_params = params[joint]
        # initial_params = get_joint_params(calibration_config.default_xml, {joint: joint_params})[joint]
        if joint_idx == 0:
            initial_params = [0.5/200, 0.1, 100/200, 1.5/10]
        if joint_idx == 1:
            initial_params = [0.3/200, 0.1, 5/200, 5.5/10]
        if joint_idx == 2:
            initial_params = [0.07/200, 0.1, 0.3/200, 0.65/10]
        if joint_idx == 3:
            initial_params = [1.5/200, 0.1, 0.7/200, 0.27/10]
        if joint_idx == 4:
            initial_params = [0.55/200, 0.1, 0.005/200, 0.035/10]
        if joint_idx == 5:
            initial_params = [100/200, 0.1, 100/200, 0.1/10]
        if joint_idx == 6:
            initial_params = [0.39/200, 0.1, 0.05/200, 0.01/10]
        # initial_params = [initial_params[p] for p in joint_params]

        def f(data, *paramvals):
            env = build_env()
            damping = copy.copy(env._wrapped_env.model.dof_damping)
            stiffness = copy.copy(env._wrapped_env.model.jnt_stiffness)
            armature = copy.copy(env._wrapped_env.model.dof_armature)
            frictionloss = copy.copy(env._wrapped_env.model.dof_frictionloss)

            damping[joint_idx] = paramvals[0]
            frictionloss[joint_idx] = paramvals[1]
            armature[joint_idx] = paramvals[2]
            stiffness[joint_idx] = paramvals[3]

            env._wrapped_env.model.dof_damping = damping * 200
            env._wrapped_env.model.dof_frictionloss = frictionloss * 1
            env._wrapped_env.model.dof_armature = armature * 200
            env._wrapped_env.model.jnt_stiffness = stiffness * 10

            ret = rollouts(env, data['ctrl'], np.stack(data['qpos'])[:, 0, :])
            ret = ret[:, :, joint_idx].flatten()
            return ret
        target = np.stack(train_data[joint]['qpos'])[:, :, joint_idx].flatten()
        print(target)

        bounds = (np.zeros(len(initial_params)), np.ones(len(initial_params)))
        optimized_params, _ = scipy.optimize.curve_fit(f, train_data[joint], target, p0=initial_params, bounds=bounds, verbose=2)

        result.update({
            joint: dict([(param, paramval) for param, paramval in zip(joint_params, optimized_params)])})
        print({joint: dict([(param, paramval) for param, paramval in zip(joint_params, optimized_params)])})
        print({joint: dict([(param, paramval) for param, paramval in zip(joint_params, optimized_params)])})

    print('Calibrated Results')
    print(result)

    return result

def mat2euler(M):
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > 1e-3: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else:
        assert(0)
    return x, y, z

def euler2mat(vec):
    x, y, z = vec[0], vec[1], vec[2]
    Ms = []
    cosz = math.cos(z)
    sinz = math.sin(z)
    Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    cosy = math.cos(y)
    siny = math.sin(y)
    Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    cosx = math.cos(x)
    sinx = math.sin(x)
    Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    return functools.reduce(np.dot, Ms[::-1])

