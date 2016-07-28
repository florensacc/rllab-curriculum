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
    STEPS_PER_EPOCH = 250000
    EPOCHS = 100
    STEPS_PER_TEST = 10000
    EXPERIMENT_PREFIX = "data/local/deep_q_rl/"
    EXPERIMENT_DIRECTORY = None

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "sandbox/haoran/deep_q_rl/roms/"
    ROM = 'freeway.bin'
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
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'crop'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False
    USE_DOUBLE = False
    CLIP_REWARD = True
    USE_BONUS = True

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
