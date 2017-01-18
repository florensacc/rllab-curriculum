"""
Specific to MultiGoalEnv.
Load snapshots and plot the q values at given points.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rllab.misc.instrument import query_yes_no
from rllab.misc.console import colorize
from sandbox.haoran.mddpg.algos.mddpg import MDDPG
from sandbox.haoran.myscripts.myutilities import get_true_env
from sandbox.haoran.mddpg.misc.sampler import MNNParallelSampler
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
from sandbox.haoran.mddpg.policies.mnn_policy import MNNPolicy

def eval_critic(sess, qf, o, lim):
    xx = np.arange(-lim, lim, 0.05)
    X, Y = np.meshgrid(xx, xx)
    all_actions = np.vstack([X.ravel(), Y.ravel()]).transpose()
    obs = np.array([o] * all_actions.shape[0])

    feed = {
        qf.observations_placeholder: obs,
        qf.actions_placeholder: all_actions
    }
    Q = sess.run(qf.output, feed).reshape(X.shape)
    return X, Y, Q



def plot_for_one_file(pkl_file, img_file):
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
            raise NotImplementedError


        lim = 1.
        # Set up all critic plots.
        critic_fig = plt.figure(figsize=(20, 7))
        ax_critics = []
        for i in range(3):
            ax = critic_fig.add_subplot(130 + i + 1)
            ax_critics.append(ax)
            plt.axis('equal')
            ax.set_xlim((-lim, lim))
            ax.set_ylim((-lim, lim))


        obss = np.array([[-2.5, 0.0],
                         [0.0, 0.0],
                         [2.5, 2.5]])

        for ax_critic, obs in zip(ax_critics, obss):

            X, Y, Q = eval_critic(sess, algo.qf, obs, lim)

            ax_critic.clear()
            cs = ax_critic.contour(X, Y, Q, 20)
            ax_critic.clabel(cs, inline=1, fontsize=10, fmt='%.0f')

            # sample and plot actions
            if isinstance(algo.policy, StochasticNNPolicy):
                all_obs = np.array([obs] * algo.K)
                all_actions = algo.policy.get_actions(all_obs)[0]
            elif isinstance(algo.policy, MNNPolicy):
                all_actions, info = algo.policy.get_action(obs, k="all")
            else:
                raise NotImplementedError

            x = all_actions[0][:, 0]
            y = all_actions[0][:, 1]
            ax_critic.plot(x, y, '*')

        # write down the hyperparams in the title of the first axis
        variant_file = os.path.join(
            os.path.dirname(pkl_file),
            "variant.json",
        )
        with open(variant_file) as vf:
            variant = json.load(vf)
        snapshot_name = os.path.basename(pkl_file).split('.pkl')[0]
        fig_title = variant["exp_name"] + "\n" + snapshot_name + " visit \n"
        for key, value in variant.items():
            if key in args.keys:
                fig_title += "%s: %s \n"%(key, value)

        # other axes will be the observation point
        for i in range(len(ax_critics)):
            ax_title = "state: (%.2f, %.2f)"%(obss[i][0], obss[i][1])
            if i == 0:
                ax_title = fig_title + ax_title
            ax_critics[i].set_title(ax_title, multialignment="left")
        critic_fig.tight_layout()

        # save to file
        plt.savefig(img_file)
        plt.cla()
        plt.close('all')
        gc.collect()

def get_img_file_name(pkl_file):
    snapshot_name = os.path.basename(pkl_file).split('.pkl')[0]
    img_file = os.path.join(
        os.path.dirname(pkl_file),
        snapshot_name + "_qf.png",
    )
    return img_file


parser = argparse.ArgumentParser()
parser.add_argument('prefix', type=str, default='??????',nargs='?', help="""
    e.g. "../exp-000/exp-000_swimmer" will operate on all swimmer folders,
    Alternatively, use ""../exp-000/". The final "/" is necessary.
""")
parser.add_argument('--max_path_length', type=int, default=-1)
parser.add_argument('--postfix', type=str, default=".pkl")
parser.add_argument('--no-skip', default=False, action='store_true')
parser.add_argument('--keys', type=str, nargs='*',
    default=["K", "alpha", "q_target_type", "ou_sigma", "freeze_samples"])
parser.add_argument('--pkl', type=str, nargs='?', default='',
    help="Specify a single .pkl file to plot.")
args = parser.parse_args()


if args.pkl != '':
    img_file = get_img_file_name(args.pkl)
    plot_for_one_file(args.pkl, img_file)
else:
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
        img_file = get_img_file_name(pkl_file)
        if os.path.exists(img_file) and not args.no_skip:
            print("Skipping", pkl_file)
            continue
        else:
            print("Processing", pkl_file)
        plot_for_one_file(pkl_file, img_file)
