import sys
import argparse
import types
from pydoc import locate

import pygame

from rllab.mdp.base import MDP
from rllab.misc.resolve import load_class
import time
import numpy as np


def sample_action(lb, ub):
    Du = len(lb)
    if np.any(np.isinf(lb)) or np.any(np.isinf(ub)):
        raise ValueError('Cannot sample unbounded actions')
    return np.random.rand(Du) * (ub - lb) + lb

def visualize_mdp(mdp, mode, max_steps=sys.maxint, fps=20):
    # step ahead with all-zero action
    delay = 1.0 / fps
    viewer = mdp.start_viewer()
    if mode == 'noop':
        action = np.zeros(mdp.action_dim)
        state = mdp.reset()[0]
        mdp.plot()
        for _ in xrange(max_steps):
            state, _, _, done = mdp.step(state, action)
            mdp.plot()
            time.sleep(delay)
            if done:
                state = mdp.reset()[0]
    elif mode == 'random':
        state = mdp.reset()[0]
        lb, ub = mdp.action_bounds
        totrew = 0
        mdp.plot()
        for _ in xrange(max_steps):
            action = sample_action(lb, ub)
            state, _, rew, done = mdp.step(state, action)
            mdp.plot()
            totrew += rew
            print rew
            if done:
                print totrew
                totret = 0
                state = mdp.reset()[0]
        if not done:
            print totrew
    elif mode == 'static':
        mdp.reset()
        while True:
            mdp.plot()
            time.sleep(delay)
    elif mode == 'human':
        state = mdp.reset()[0]
        mdp.plot()
        tr = 0.
        for _ in xrange(max_steps):
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            action = mdp.action_from_keys(keys)
            state, _, r, done = mdp.step(state, action)
            tr += r
            mdp.plot()
            time.sleep(delay)
            if done:
                print "Episode done, reward: ", tr
                tr = 0.
                state = mdp.reset()[0]
    else:
        raise ValueError('Unsupported mode: %s' % mode)
    mdp.stop_viewer()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str, required=True, help='module path to the mdp class')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['noop', 'random', 'static', 'human'],
                        help='module path to the mdp class')
    parser.add_argument('--fps', type=int, default=20, help='frames per second')
    parser.add_argument('--max_steps', type=int, default=sys.maxint, help='max steps')
    args = parser.parse_args()
    mdp = load_class(args.mdp, MDP, ["rllab", "mdp"])()#load_mdp_class(args.mdp)()
    visualize_mdp(mdp, mode=args.mode, max_steps=args.max_steps, fps=args.fps)
