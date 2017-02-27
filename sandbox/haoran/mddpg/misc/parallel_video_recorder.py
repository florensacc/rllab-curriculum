from rllab.misc.console import colorize
from rllab.misc.ext import set_seed
from sandbox.haoran.myscripts.myutilities import get_true_env
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import joblib
import time
import pyprind

class ParallelVideoRecorder(object):
    """
    Runs several simulations in parallel, specifically for MuJoCo.
    This file loads pickled snapshots with tensorflow sessions.
    """
    def __init__(
            self,
            n_parallel,
            pkl_files,
            render_configs,
            seeds,
            max_path_length, animated,
            setup_script="",
            video_config=None,
        ):
        assert all([
            len(pkl_files) == n_parallel,
            len(render_configs) == n_parallel,
        ])
        self.n_parallel = n_parallel
        self.pkl_files = pkl_files
        self.render_configs = render_configs
        self.seeds = seeds
        self.max_path_length = max_path_length
        self.animated = animated
        self.video_config = video_config
        self.recorded = (video_config is not None)
        self.setup_script = setup_script

    def _start_session(self):
        self._sess = tf.Session()
        self._sess.__enter__()

    def _close_session(self):
        self._sess.close()

    def _load_snapshot(self, pkl_file):
        self._snapshot = joblib.load(pkl_file)
        print(colorize(
            "Worker %d finished loading the snapshot"%(self._rank),
            "green",
        ))

    def _simulate(self, render_config):
        agent = self._snapshot["algo"].policy
        env = self._snapshot["algo"].env
        es = self._snapshot["algo"].exploration_strategy
        # hack
        if self._snapshot["algo"].__class__.__name__ == "DDPG":
            use_es = True
        exec(self.setup_script)
        if not use_es:
            print(self._rank)

        # set seed
        seed = self.seeds[self._rank]
        set_seed(seed)

        if self.recorded:
            img_list = []

        def render(**kwargs):
            if self.animated:
                env.render(config=render_config, **kwargs)
                time.sleep(render_config["sleep_time"])
            if self.recorded:
                img = env.render(config=render_config, mode='rgb_array', **kwargs)
                img_list.append(img)


        o = env.reset()
        agent.reset()
        if use_es:
            es.reset()
        t = 0
        rewards = []
        render()
        while t < self.max_path_length:
            if use_es:
                a = es.get_action(t, o, agent)
            else:
                a, _ = agent.get_action(o)
            next_o, reward, done, env_info = env.step(a)
            t += 1
            o = next_o
            render()
            # do not reset yet
        render(close=True)

        if self.recorded:
            self._shared_dict["img_list_%d"%(self._rank)] = img_list

    def _run(self, rank, shared_dict):
        self._rank = rank
        self._shared_dict = shared_dict
        self._start_session()
        self._load_snapshot(self.pkl_files[self._rank])
        self._simulate(
            render_config=self.render_configs[self._rank]
        )
        self._close_session()

    def run(self):
        if self.n_parallel == 1:
            self.shared_dict = dict()
            self._run(rank=0, shared_dict=self.shared_dict)
        else:
            manager = mp.Manager()
            self.shared_dict = manager.dict()
            processes = [
                mp.Process(target=self._run, args=(rank, self.shared_dict))
                for rank in range(self.n_parallel)
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
        if self.recorded:
            self.generate_video(self.video_config)

    def generate_video(self, video_config):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        wargs = video_config["window_args"]
        height = wargs["height"]
        width = wargs["width"]
        n_row = wargs["n_row"]
        n_col = wargs["n_col"]
        x_margin = wargs["x_margin"]
        y_margin = wargs["y_margin"]

        # set up the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.set_size_inches(video_config["figsize"])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # no fig margins

        # initialize the image
        combined_img = np.zeros(
            ((height + y_margin) * (n_row-1) + height,
             (width + x_margin) * (n_col-1) + width,
             3),
            dtype=np.uint8,
        )
        I = ax.imshow(combined_img)
        combined_img_list = []


        all_img_lists = [
            self.shared_dict["img_list_%d"%(rank)]
            for rank in range(self.n_parallel)
        ]

        for t in range(self.max_path_length):
            for rank in range(self.n_parallel):
                img = all_img_lists[rank][t]
                cur_row = int(np.floor(rank / n_col))
                cur_col = np.mod(rank, n_col)
                xpos = (width + x_margin) * cur_col
                ypos = (height + y_margin) * cur_row
                combined_img[ypos: ypos + height, xpos: xpos + width, :] = img
            combined_img_list.append(np.copy(combined_img))

        progbar = pyprind.ProgBar(
            self.max_path_length,
            monitor=True,
            title=colorize('\nWriting video', "yellow"),
            bar_char='█',
        )
        def update_img(img):
            I.set_data(img)
            progbar.update()
        ani = animation.FuncAnimation(fig, update_img, combined_img_list)
        writer = animation.writers['ffmpeg'](fps=video_config["fps"])
        ani_file = video_config["output"]
        ani.save(ani_file, writer=writer, dpi=video_config["dpi"])
        print(colorize("Video written to %s"%(ani_file), "green"))



def generate_window_configs(
        n_window,
        width, height,
        n_row, n_col,
        x_start, y_start,
        x_margin, y_margin
    ):
    window_configs = []
    for i in range(n_window):
        cur_row = int(np.floor(i / n_col))
        cur_col = np.mod(i, n_col)

        config = dict()
        config["xpos"] = (width + x_margin) * cur_col + x_start
        config["ypos"] = (height + y_margin) * cur_row + y_start
        config["width"] = width
        config["height"] = height
        window_configs.append(config)
    return window_configs



# testing -----------------------------------------------------------------
# Run the pretrained environment in parallel with different random seeds

if __name__  == "__main__":
    max_path_length = 500
    pkl_file = "/Users/haoran/Academics/RL/mddpg/rllab-private/data/s3/mddpg/vddpg/ant/exp-004b/exp-004b_20170218_223241_297225_tuomas_ant/itr_499.pkl"
    n_parallel = 2
    pkl_files = [pkl_file for i in range(n_parallel)]
    seeds = list(np.arange(0, n_parallel * 100, 100))

    window_args = dict(
        n_window=n_parallel,
        width=500, height=500,
        n_row=2, n_col=4,
        x_start=1000, y_start=0,
        x_margin=20, y_margin=20,
    )
    render_configs = generate_window_configs(**window_args)
    for config, seed in zip(render_configs, seeds):
        config["title"] = "seed: %d"%(seed)
        config["sleep_time"] = 0.005

    setup_script = """
from sandbox.haoran.myscripts.myutilities import get_true_env
true_env = get_true_env(env)
viewer = true_env.get_viewer()
viewer.cam.trackbodyid = -1 # fix camera at the origin
viewer.cam.elevation = -50 # overhead
viewer.cam.distance = 10
    """

    video_config = dict(
        output = "/Users/haoran/Google Drive/2016-17 2nd/vddpg/figure/pretraining/pretrained/exp-004b_20170218_223241_297225_tuomas_ant_itr_499.mp4",
        window_args=window_args,
        fps=50,
        figsize=(window_args["n_col"] * 5, window_args["n_row"] * 5),
        dpi=100,
    )

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
