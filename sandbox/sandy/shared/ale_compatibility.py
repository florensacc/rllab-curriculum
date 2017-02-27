import os, sys

from gym.utils import seeding
from sandbox.sandy.misc.util import suppress_stdouterr

def set_gym_seed(gym_env, seed=None):
    # Near-duplicate of OpenAI Gym atari_env._seed method
    # However, loads ROM correctly for both atari_py ALE interface (what Gym
    # uses) and original ALE interface
    # (Not changing this directly in gym module, to preserve Docker compatibility)

    gym_env.np_random, seed1 = seeding.np_random(seed)
    # Derive a random seed. This gets passed as a uint, but gets
    # checked as an int elsewhere, so we need to keep it below
    # 2**31.
    seed2 = seeding.hash_seed(seed1 + 1) % 2**31
    # Empirically, we need to seed before loading the ROM.
    gym_env.ale.setInt(b'random_seed', seed2)

    if 'atari_py' in str(type(gym_env.ale)):
        with suppress_stdouterr():
            gym_env.ale.loadROM(gym_env.game_path)
    else:
        with suppress_stdouterr():
            gym_env.ale.loadROM(str.encode(gym_env.game_path))
    return [seed1, seed2]
