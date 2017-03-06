"""
Designed for ant/exp-004b, in which the path plotting was wrong
Grab the .pkl files, load the algo, plot the paths, and then output to files.
"""
from sandbox.haoran.myscripts.retrainer import Retrainer
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.console import colorize
from rllab import config
from rllab.viskit.core import flatten_dict

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json


exp_prefix="mddpg/vddpg/ant/exp-004b"
img_name = "env_itr_00089_replot.png"
interested_keys = [
    "scale_reward",
    "reward_type",
    "train_frequency.update_target_frequency"
]

# find the exp paths
paths = []
dirname = "data/s3/%s"%(exp_prefix)
for subdirname in os.listdir(dirname):
    path = os.path.join(dirname,subdirname)
    if os.path.isdir(path):
        paths.append(path)

inputs = []
for path in paths:
    variant_file = os.path.join(path, "variant.json")
    image_file = os.path.join(path, img_name)
    if os.path.exists(variant_file) and os.path.exists(image_file):
        print(colorize(
            "Showing %s"%(image_file), "yellow",
        ))
        # show the variants
        with open(variant_file) as vf:
            v = json.load(vf)
        for k in interested_keys:
            v_flat = flatten_dict(v)
            print(k, ": ", v_flat[k])

        # show the image
        img = plt.imread(image_file)
        plt.imshow(img)
        plt.pause(0.05)

        input("Press Enter to continue...")
