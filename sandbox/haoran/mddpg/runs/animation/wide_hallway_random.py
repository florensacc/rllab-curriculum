"""
Goal: visualize the policy across seeds and training epochs.

For debugging, tune the following to reduce computation time:
    max_path_length, n_parallel, dpi, fps
    recorded = False

Choose good random seeds to illustrate the effect
"""

"""
Recommended changes to vendor/mujoco_models/ant.xml:
    <material name='MatPlane' texture="texplane" shininess="1" texrepeat="20 20" specular="1"  reflectance="0.5" />
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0" rgb2="0 0 0" width="100" height="100" />
"""
from sandbox.haoran.mddpg.misc.parallel_video_recorder import \
    ParallelVideoRecorder, generate_window_configs
import os
import numpy as np
import tensorflow as tf

def run(pkl_file, output, recorded, seeds):
    max_path_length = 500
    n_parallel = 1
    pkl_files = [pkl_file for i in range(n_parallel)]

    window_args = dict(
        n_window=n_parallel,
        width=500, height=500,
        n_row=1, n_col=1,
        x_start=1000, y_start=0,
        x_margin=20, y_margin=20,
    )
    render_configs = generate_window_configs(**window_args)
    for config, seed in zip(render_configs, seeds):
        config["title"] = "seed: %d"%(seed)
        config["sleep_time"] = 0.00001

    setup_script = """
from sandbox.haoran.myscripts.myutilities import get_true_env
true_env = get_true_env(env)
viewer = true_env.get_viewer()
viewer.cam.trackbodyid = -1 # fix camera at the origin
viewer.cam.elevation = -50 # overhead
viewer.cam.azimuth = 190
viewer.cam.distance = 20
    """

    if recorded:
        video_config = dict(
            output=output,
            window_args=window_args,
            fps=50,
            figsize=(window_args["n_col"] * 5, window_args["n_row"] * 5),
            dpi=400,
        )
    else:
        video_config = None

    vr = ParallelVideoRecorder(
        n_parallel=n_parallel,
        pkl_files=pkl_files,
        render_configs=render_configs,
        seeds=seeds,
        max_path_length=max_path_length,
        animated=True,
        video_config=video_config,
        setup_script=setup_script,
    )
    vr.run()

if __name__ == "__main__":
    good_seeds = np.arange(10)
    exp_prefix = "mddpg/vddpg/ant/exp-004f"
    exp_name = "exp-004f_20170220_212252_502988_ant_puddle"
    itr = 0
    # itr = 99
    # itr = 199
    # has it converged yet?

    path = os.path.join(
        "data/s3",
        exp_prefix,
        exp_name
    )
    pkl_file = os.path.join(path, "itr_%d.pkl"%(itr))
    output_path = "/Users/haoran/Google Drive/2016-17 2nd/vddpg/figure/pretraining/wide_hallway/%s_itr_%d"%(
        exp_name, itr
    )
    os.system("mkdir -p \"%s\""%(output_path))
    for i in good_seeds:
        tf.reset_default_graph()
        sess = tf.get_default_session()
        if sess is not None:
            sess.close()
        output = os.path.join(
            output_path,
            "%03d.mp4"%(i),
        )
        print(output)
        run(
            pkl_file=pkl_file,
            output=output,
            recorded=True,
            seeds=[i],
        )
