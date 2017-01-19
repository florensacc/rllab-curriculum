"""
Load a snapshot (specific to MDDPG), sample trajs, and plot the visitation map.
TODO:
1. use parallel sampling (beware of setting different heads for different
samplers)
"""


import argparse
import os
import csv
import numpy as np
import joblib
import sys
import gc
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize
from sandbox.haoran.mddpg.algos.mddpg import MDDPG
from sandbox.haoran.myscripts.myutilities import get_true_env
from sandbox.haoran.mddpg.misc.sampler import MNNParallelSampler

parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?', help="""
    e.g. "../exp-000/exp-000_swimmer" will operate on all swimmer folders,
    Alternatively, use ""../exp-000/". The final "/" is necessary.
""")
parser.add_argument('--max_path_length', type=int, default=-1)
parser.add_argument('--mesh_density', type=int, default=50)
parser.add_argument('--postfix', type=str, default=".pkl")
parser.add_argument('--no-skip', default=False, action='store_true')
parser.add_argument('--keys', type=str, nargs='*',
default=["K", "alpha"])
args = parser.parse_args()


pkl_files = []
dirname = os.path.dirname(args.prefix) # ../exp-000
subdirprefix = os.path.basename(args.prefix) # exp-000_swimmer
for subdirname in os.listdir(dirname): # exp-000_swimmer_20170101
    path = os.path.join(dirname,subdirname) # ../exp-000/exp-000_swimmer_20170101
    if os.path.isdir(path) and (subdirname.startswith(subdirprefix)):
        # non-empty and starts with the prefix exp-000_swimmer
        for filename in os.listdir(path):
            if filename.endswith(args.postfix):
                pkl_file = os.path.join(path, filename)
                pkl_files.append(pkl_file)

for pkl_file in pkl_files:
    # skip file if it the image already exists
    snapshot_name = os.path.basename(pkl_file).split('.pkl')[0]
    img_file = os.path.join(
        os.path.dirname(pkl_file),
        snapshot_name + "_visit.png",
    )
    if os.path.exists(img_file) and not args.no_skip:
        print("Skipping", pkl_file)
        continue
    else:
        print("Processing", pkl_file)

    tf.reset_default_graph() # avoid repeated tensor names
    with tf.Session() as sess:
        # read data
        snapshot = joblib.load(pkl_file)
        if "algo" in snapshot:
            algo = snapshot["algo"]
            max_path_length = algo.max_path_length
            K = algo.K
            env = algo.env
        else:
            if args.max_path_length == -1:
                print("""
                    Cannot find max_path_length from the snapshot.
                    Please specify it.
                """)
                sys.exit(1)
            else:
                assert args.max_path_length > 0
                max_path_length = args.max_path_length
            K = snapshot["policy"].K
            algo = MDDPG(
                env=snapshot["env"],
                exploration_strategy=snapshot["es"],
                policy=snapshot["policy"],
                kernel=snapshot["kernel"],
                qf=snapshot["qf"],
                K=K,
                max_path_length=max_path_length,
                resume=True, # very important
                # other training params are irrelevant here
            )
            env = snapshot["env"]

        # sample paths
        algo.eval_sampler = MNNParallelSampler(algo)
        algo._start_worker()
        paths = algo.eval_sampler.obtain_samples(
            itr=0,
            max_path_length=max_path_length,
            max_head_repeat=1,
        )

        # plotting
        fig, ax = plt.subplots()
        true_env = get_true_env(env)
        true_env.plot_visitation(
            epoch=0,
            paths=paths,
            mesh_density=args.mesh_density,
            fig=fig,
            ax=ax,
        )

        # write down the hyperparams in the title
        variant_file = os.path.join(
            os.path.dirname(pkl_file),
            "variant.json",
        )
        with open(variant_file) as vf:
            variant = json.load(vf)
        title = variant["exp_name"] + "\n" + snapshot_name + " visit \n"
        for key, value in variant.items():
            if key in args.keys:
                title += "%s: %s \n"%(key, value)
        ax.set_title(title, multialignment="left")
        fig.tight_layout()

        # save to file
        plt.savefig(img_file)
        plt.cla()
plt.close('all')
gc.collect()
