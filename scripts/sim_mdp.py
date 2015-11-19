import sys
import argparse
import types
from pydoc import locate
from rllab.mdp.base import MDP
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
        mdp.plot()
        for _ in xrange(max_steps):
            action = sample_action(lb, ub)
            state, _, _, done = mdp.step(state, action)
            mdp.plot()
            time.sleep(delay)
            if done:
                state = mdp.reset()[0]
    elif mode == 'static':
        mdp.reset()
        while True:
            mdp.plot()
            time.sleep(delay)
    else:
        raise ValueError('Unsupported mode: %s' % mode)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str, required=True, help='module path to the mdp class')
    parser.add_argument('--mode', type=str, default='static', choices=['noop', 'random', 'static'], help='module path to the mdp class')
    parser.add_argument('--fps', type=int, default=20, help='frames per second')
    parser.add_argument('--max-steps', type=int, default=sys.maxint, help='max steps')
    args = parser.parse_args()
    mdp = load_mdp_class(args.mdp)()
    visualize_mdp(mdp, mode=args.mode, max_steps=args.max_steps, fps=args.fps)
