import argparse
import sys
import time

import numpy as np
import pygame

from rllab.mdp.base import MDP
from rllab.misc.resolve import load_class


def sample_action(lb, ub):
    Du = len(lb)
    if np.any(np.isinf(lb)) or np.any(np.isinf(ub)):
        raise ValueError('Cannot sample unbounded actions')
    return np.random.rand(Du) * (ub - lb) + lb


def to_onehot(ind, dim):
    ret = np.zeros(dim)
    ret[ind] = 1
    return ret


def visualize_mdp(mdp, mode, max_steps=sys.maxint, speedup=1):
    # step ahead with all-zero action
    if mode == 'noop':
        action = np.zeros(mdp.action_dim)
        mdp.reset()
        mdp.plot()
        for _ in xrange(max_steps):
            _, _, done = mdp.step(action)
            mdp.plot()
            time.sleep(mdp.timestep / speedup)
            if done:
                mdp.reset()
    elif mode == 'random':
        mdp.reset()
        if np.dtype(mdp.action_dtype).kind in ['i', 'u']:
            sampler = lambda: to_onehot(
                np.random.choice(mdp.action_dim), mdp.action_dim)
        else:
            lb, ub = mdp.action_bounds
            sampler = lambda: sample_action(lb, ub)
        totrew = 0
        mdp.plot()
        for i in xrange(max_steps):
            action = sampler()
            _, rew, done = mdp.step(action)
            # if i % 10 == 0:
            mdp.plot()
            # import time as ttime
            time.sleep(mdp.timestep / speedup)
            totrew += rew
            if done:
                totrew = 0
                mdp.reset()
        if not done:
            totrew = 0
    elif mode == 'static':
        mdp.reset()
        while True:
            mdp.plot()
            time.sleep(mdp.timestep / speedup)
    elif mode == 'human':
        mdp.reset()
        mdp.plot()
        tr = 0.
        from rllab.env.box2d.box2d_mdp import Box2DEnv
        if isinstance(mdp, Box2DEnv):
            for _ in xrange(max_steps):
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                action = mdp.action_from_keys(keys)
                ob, r, done = mdp.step(action)
                tr += r
                mdp.plot()
                time.sleep(mdp.timestep / speedup)
                if done:
                    tr = 0.
                    mdp.reset()
        else:
            trs = [tr]
            actions = [np.zeros(2)]
            from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
            from rllab.mdp.mujoco_1_22.gather.gather_mdp import GatherMDP
            from rllab.mdp.mujoco_1_22.maze_mdp import MazeMDP
            if isinstance(mdp, (MujocoMDP, GatherMDP, MazeMDP)):
                from rllab.mjcapi.rocky_mjc_1_22 import glfw

                def cb(window, key, scancode, action, mods):
                    actions[0] = mdp.action_from_key(key)
                glfw.set_key_callback(mdp.viewer.window, cb)
                while True:
                    try:
                        actions[0] = np.zeros(2)
                        glfw.poll_events()
                        # if np.linalg.norm(actions[0]) > 0:
                        ob, r, done = mdp.step(actions[0])
                        trs[0] += r
                        mdp.plot()
                        time.sleep(mdp.timestep / speedup)
                        if done:
                            trs[0] = 0.
                            mdp.reset()
                    except Exception as e:
                        print e
    else:
        raise ValueError('Unsupported mode: %s' % mode)
    mdp.stop_viewer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mdp', type=str,
                        help='module path to the mdp class')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['noop', 'random', 'static', 'human'],
                        help='module path to the mdp class')
    parser.add_argument('--speedup', type=float, default=1, help='speedup')
    parser.add_argument('--max_steps', type=int,
                        default=sys.maxint, help='max steps')
    args = parser.parse_args()
    mdp = load_class(args.mdp, MDP, ["rllab", "mdp"])()
    visualize_mdp(mdp, mode=args.mode, max_steps=args.max_steps,
                  speedup=args.speedup)
