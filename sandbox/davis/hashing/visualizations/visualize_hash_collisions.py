import argparse
import joblib
import uuid
import os
import random
import numpy as np
import tensorflow as tf
import itertools

from collections import defaultdict

THRESHOLD = 0  # Figure out what this should be later

filename = str(uuid.uuid4())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    with tf.Session():
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        bonus_evaluator = data['bonus_evaluator']
        paths = data['paths']
        keys_to_states = defaultdict(list)
        for path in paths:
            states = path['observations']
            for state in states:
                keys = bonus_evaluator.compute_keys(state)
                for key in keys:
                    keys_to_states[key].append(state)

        bad_pairs = []
        for key, states in keys_to_states.items():
            for state1, state2 in itertools.combinations(states, 2):
                if np.linalg.norm(state1 - state2) > THRESHOLD:
                    bad_pairs.append((state1, state2, key))

        for _ in range(20):
            env.render()  # Get through the initial zoom-out

        bad_pairs.sort(key=lambda s: -np.linalg.norm(s[0] - s[1]))
        for state1, state2, key in bad_pairs[:5]:
            env.reset()
            env.wrapped_env.set_state_tmp(state1, restore=False)
            env.render()
            input(key)
            env.reset()
            env.wrapped_env.set_state_tmp(state2, restore=False)
            env.render()
            input(key)
