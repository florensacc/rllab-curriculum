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

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 300
    EPOCHS = 2
    STEPS_PER_TEST = 300
    EXPERIMENT_DIRECTORY = None # use default, see launcher.py
    EXPERIMENT_PREFIX = "data/local/deep_q_rl/"

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "sandbox/haoran/deep_q_rl/roms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    BATCH_ACCUMULATOR = 'mean'
    LEARNING_RATE = .0002
    DISCOUNT = .95
    RMS_DECAY = .99 # (Rho)
    RMS_EPSILON = 1e-6
    MOMENTUM = 0
    CLIP_DELTA = 0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nips_dnn"
    FREEZE_INTERVAL = 1
    REPLAY_START_SIZE = 100
    RESIZE_METHOD = 'crop'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'false'
    MAX_START_NULLOPS = 0
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False
    USE_DOUBLE = True
    CLIP_REWARD = True
    USE_BONUS = True
    AGENT_UNPICKLABLE_LIST = ["data_set","test_data_set"]

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
