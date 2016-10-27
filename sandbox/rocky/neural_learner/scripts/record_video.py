import os.path
import os
import joblib
from rllab import config
import numpy as np
import random
from gym.monitoring.video_recorder import ImageEncoder
import tensorflow as tf
import cv2
from rllab.misc import console
from rllab.misc import instrument
from sandbox.rocky.analogy.utils import unwrap
import subprocess


def to_img(obs):
    return cv2.resize(np.cast['uint8']((obs / 2 + 0.5) * 255.0), (100, 100))


print(__file__)

if not os.path.exists('/.dockerenv'):
    # first run in docker
    docker_command = instrument.to_docker_command(
        params=dict(log_dir="/dev/null"),
        docker_image="dementrock/rllab3-vizdoom-gpu-cuda80",
        dry=True,
        use_gpu=False,
        script=__file__,
        post_commands=[],
    )
    print(docker_command)
    p = subprocess.Popen(docker_command, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            print("terminating")
            p.terminate()
        except OSError:
            print("os error!")
            pass
        p.wait()
else:
    with tf.Session() as sess:
        np.random.seed(0)
        random.seed(0)

        pkl_file = os.path.join(config.PROJECT_PATH,
                                "data/local/doom-maze-2/doom_maze_2_2016_09_26_10_58_35_0001/params.pkl")
        output_path = os.path.join(config.PROJECT_PATH, "data/video/doom-maze-2")
        console.mkdir_p(output_path)

        data = joblib.load(pkl_file)

        policy = data["policy"]
        env = data["env"]

        for idx in range(10):
            print("Generating video %d.mp4" % idx)
            encoder = ImageEncoder(
                output_path=os.path.join(output_path, '%d.mp4' % idx),
                frame_shape=(120, 160, 3),
                frames_per_sec=10
            )
            obs = env.reset()
            policy.reset()
            for t in range(500):
                action, _ = policy.get_action(obs)
                encoder.capture_frame(unwrap(env).get_image_obs(rescale=False))
                obs, _, done, _ = env.step(action)
                if done:
                    break
            encoder.close()
