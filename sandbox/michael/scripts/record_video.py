import os.path as osp
import joblib
from rllab import config
import numpy as np
import random
from gym.monitoring.video_recorder import ImageEncoder
import tensorflow as tf
import cv2
from rllab.misc import console

frame_size = (500, 500)

def to_img(obs, frame_size=(100, 100)):
    return cv2.resize(np.cast['uint8'](obs), frame_size)
    # return cv2.resize(np.cast['uint8']((obs / 2 + 0.5) * 255.0), frame_size)
    # return obs

with tf.Session() as sess:
    np.random.seed(0)
    random.seed(0)

    pkl_file = osp.join(config.PROJECT_PATH,
                        '/Users/florensacc/Library/goal-rl/rllab_goal_rl/sandbox/dave/upload/maze_best/itr_199/itr_4.pkl'
                        )
    output_path = osp.join(config.PROJECT_PATH, "data/video/goalGAN_maze")
    console.mkdir_p(output_path)

    # import pdb; pdb.set_trace()
    data = joblib.load(pkl_file)

    policy = data["policy"]

    env = data["env"]
    # env = SwimmerEnv()

    for idx in range(7, 8):

        encoder = ImageEncoder(output_path=osp.join(output_path, '%d_goalGAN_maze.mp4' % idx),
                               frame_shape=frame_size + (3,),
                               frames_per_sec=15)

        for i in range(6):
            obs = env.reset()
        print("Generating %d_goalGAN_maze.mp4" % idx)
        image = env.render(mode='rgb_array')
        policy.reset()
        for t in range(500):
            compressed_image = to_img(image, frame_size=frame_size)
            # cv2.imshow('frame{}'.format(t), compressed_image)
            cv2.waitKey(10)
            encoder.capture_frame(compressed_image)
            action, _ = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            image = env.render(mode='rgb_array')
            if done:
                break
        encoder.close()
