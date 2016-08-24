#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

"""
import sys
sys.path.append("sandbox/haoran/deep_q_rl/deep_q_rl")

import launcher
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('networkfile', nargs='?',help='Network file. Use "none" to test a newly created (ie random) network')
parser.add_argument('--game',nargs='?',help="Name of the game .bin file",default="breakout")
parser.add_argument('--eps',nargs='?',help="Random exploration param.",default=0.)
parser.add_argument('--steps',nargs='?',help="Number of total time steps to watch (across different episodes)",default=10000)
args,unknowns = parser.parse_known_args(sys.argv[1:])


class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = args.steps
    EPOCHS = 10
    STEPS_PER_TEST = 1
    EXPERIMENT_DIRECTORY = None # use default, see launcher.py
    EXPERIMENT_PREFIX = "data/local/deep_q_rl/"

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "sandbox/haoran/deep_q_rl/roms/"
    ROM = args.game + ".bin"
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    BATCH_ACCUMULATOR = 'mean'
    LEARNING_RATE = 0
    DISCOUNT = .95
    RMS_DECAY = .99 # (Rho)
    RMS_EPSILON = 1e-6
    MOMENTUM = 0
    CLIP_DELTA = 0
    EPSILON_START = args.eps
    EPSILON_MIN = 0.
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000
    BATCH_SIZE = 32
    NETWORK_TYPE = "linear"
    FREEZE_INTERVAL = -1
    REPLAY_START_SIZE = 100
    RESIZE_METHOD = 'crop'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 0
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False
    USE_DOUBLE = False
    CLIP_REWARD = False



command = ['--nn-file',args.networkfile,'--display-screen']
# command = ['--nn-file',args.networkfile]
launcher.launch(command, Defaults, __doc__)
