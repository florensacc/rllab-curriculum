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
import numpy as np
import os
import json


exp_prefix="mddpg/vddpg/ant/exp-004b"
img_name = "env_itr_00499_replot.png"
def is_good_variant(v):
    return all([
        v["train_frequency"]["update_target_frequency"] == 1000,
        v["reward_type"] == "velocity"
    ])
variant_list = [0.1, 1, 10, 100]
variant_tops = [0, 0, 0, 0]
n_seed = 5
# title = "dist_from_ori, 1k, \n scale_reward = 0.1, 1, 10, 100"
title = ""

# find the exp paths
paths = []
dirname = "data/s3/%s"%(exp_prefix)
for subdirname in os.listdir(dirname):
    path = os.path.join(dirname,subdirname)
    if os.path.isdir(path):
        paths.append(path)

output_file = os.path.join(dirname, "summary_00499_replot.png")

results = []
for path in paths:
    variant_file = os.path.join(path, "variant.json")
    image_file = os.path.join(path, img_name)
    if os.path.exists(variant_file) and os.path.exists(image_file):
        with open(variant_file) as vf:
            v = json.load(vf)
        if is_good_variant(v):
            results.append(
                (v["exp_name"], v["scale_reward"], plt.imread(image_file))
            )
n_variant = len(variant_list)
w, h, c = results[0][2].shape
I = np.zeros((w * n_seed, h * n_variant, c))
fig = plt.figure(figsize=(n_variant, n_seed))
for exp_name, variant, img in results:
    col = variant_list.index(variant)
    row = variant_tops[col]
    I[row * h : (row + 1) * h, col * w : (col + 1) * w, :] = img
    variant_tops[col] = variant_tops[col] + 1

    y = (row + 1) * h + 10
    x = col * w
    text = exp_name
    plt.text(x, y, text, fontsize=1.5)
plt.title(title)
plt.axis('off')
plt.tight_layout()
plt.imshow(I)
fig.savefig(output_file, dpi=600, bbox_inches='tight')
print("Output to ", output_file)
