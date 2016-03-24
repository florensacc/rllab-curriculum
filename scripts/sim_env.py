import argparse
import sys
import time

import numpy as np
import pygame

from rllab.envs.base import Env
# from rllab.env.base import MDP
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


def visualize_env(env, mode, max_steps=sys.maxint, speedup=1):
    # step ahead with all-zero action
    if mode == 'noop':
        action = np.zeros(env.action_dim)
        env.reset()
        env.render()
        for _ in xrange(max_steps):
            _, _, done = env.step(action)
            env.render()
            time.sleep(env.timestep / speedup)
            if done:
                env.reset()
    elif mode == 'random':
        env.reset()
        if np.dtype(env.action_dtype).kind in ['i', 'u']:
            sampler = lambda: to_onehot(
                np.random.choice(env.action_dim), env.action_dim)
        else:
            lb, ub = env.action_bounds
            sampler = lambda: sample_action(lb, ub)
        totrew = 0
        env.render()
        for i in xrange(max_steps):
            action = sampler()
            _, rew, done = env.step(action)
            # if i % 10 == 0:
            env.render()
            # import time as ttime
            time.sleep(0.05)#env.timestep / speedup)
            totrew += rew
            if done:
                totrew = 0
                env.reset()
        if not done:
            totrew = 0
    elif mode == 'static':
        env.reset()
        while True:
            env.render()
            time.sleep(0.05)#env.timestep / speedup)
    elif mode == 'human':
        env.reset()
        env.render()
        tr = 0.
        from rllab.envs.box2d.box2d_env import Box2DEnv
        if isinstance(env, Box2DEnv):
            for _ in xrange(max_steps):
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                action = env.action_from_keys(keys)
                ob, r, done = env.step(action)
                tr += r
                env.render()
                time.sleep(0.05)#env.timestep / speedup)
                if done:
                    tr = 0.
                    env.reset()
        else:
            trs = [tr]
            actions = [np.zeros(2)]
            from rllab.envs.mujoco.mujoco_env import MujocoEnv
            # from rllab.env.mujoco_1_22.gather.gather_env import GatherMDP
            from rllab.envs.mujoco.maze.maze_env import MazeEnv
            if isinstance(env, (MujocoEnv, MazeEnv)):#, GatherMDP, MazeMDP)):
                print "is mujoco"
                # from rllab.mjcapi.rocky_mjc_1_22 import glfw
                from rllab.mujoco_py import glfw

                def cb(window, key, scancode, action, mods):
                    actions[0] = env.action_from_key(key)
                glfw.set_key_callback(env.viewer.window, cb)
                while True:
                    try:
                        actions[0] = np.zeros(2)
                        glfw.poll_events()
                        # if np.linalg.norm(actions[0]) > 0:
                        ob, r, done, info = env.step(actions[0])
                        trs[0] += r
                        env.render()
                        # time.sleep(env.timestep / speedup)
                        time.sleep(0.05)#env.timestep / speedup)
                        if done:
                            trs[0] = 0.
                            env.reset()
                    except Exception as e:
                        print e
    else:
        raise ValueError('Unsupported mode: %s' % mode)
    # env.stop_viewer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='module path to the env class')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['noop', 'random', 'static', 'human'],
                        help='module path to the env class')
    parser.add_argument('--speedup', type=float, default=1, help='speedup')
    parser.add_argument('--max_steps', type=int,
                        default=sys.maxint, help='max steps')
    args = parser.parse_args()
    env = load_class(args.env, Env, ["rllab", "envs"])()
    visualize_env(env, mode=args.mode, max_steps=args.max_steps,
                  speedup=args.speedup)
