"""
Designed for ant/exp-004b, in which the path plotting was wrong
Grab the .pkl files, load the algo, plot the paths, and then output to files.
"""
from sandbox.haoran.myscripts.retrainer import Retrainer
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.console import colorize
from rllab import config

import tensorflow as tf
import os
import json


exp_prefix="mddpg/vddpg/ant/exp-004b"
epoch = 499
n_paths = 50
no_skip = True
configure_script = ""
def is_good_variant(v):
    return True

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
    if os.path.exists(variant_file):
        img_file = os.path.join(
            path,
            'env_itr_%05d_replot.png'%(epoch)
        )
        if os.path.exists(img_file) and not no_skip:
            print(colorize(
                "%s alredy exists. Skip."%(img_file),
                "green",
            ))
            continue

        print(colorize(
            "Processing %s"%(path), "yellow",
        ))
        # load the snapshot
        with open(variant_file) as vf:
            v = json.load(vf)
        tf.reset_default_graph()
        retrainer = Retrainer(
            exp_prefix=exp_prefix,
            exp_name=v["exp_name"],
            snapshot_file="itr_%d.pkl"%(epoch),
            configure_script=configure_script,
        )
        retrainer.local_log_dir = os.path.join(
            config.LOG_DIR,
            "s3",
            retrainer.exp_prefix.replace("_", "-"),
            retrainer.exp_name,
        )
        retrainer.reload_snapshot()
        algo = retrainer.algo

        # rollout paths
        algo._start_worker()
        paths = algo.eval_sampler.obtain_samples(
            n_paths=n_paths,
            max_path_length=algo.max_path_length,
            policy=algo.eval_policy
        )
        print("Obtained %d paths"%(len(paths)))

        # plot paths
        algo._init_figures()
        env = algo.env
        while isinstance(env, ProxyEnv):
            env = env._wrapped_env
        env.plot_paths(paths, algo._ax_env)
        algo._ax_env.set_xlim((-10,10))
        algo._ax_env.set_ylim((-10,10))

        # output
        algo._fig_env.savefig(img_file, dpi=100)
        print("Output to:", img_file)

        retrainer.sess.close()
