import os.path as osp
import joblib
from rllab import config
import numpy as np
import random
from gym.monitoring.video_recorder import ImageEncoder
import tensorflow as tf
import cv2
from rllab.misc import console


def to_img(obs):
    return cv2.resize(np.cast['uint8']((obs / 2 + 0.5) * 255.0), (100, 100))


with tf.Session() as sess:
    np.random.seed(0)
    random.seed(0)

    pkl_file = osp.join(config.PROJECT_PATH,
                        "data/local/analogy-particle-23/analogy-particle-23-5-particles/params.pkl")
    output_path = osp.join(config.PROJECT_PATH, "data/video/analogy-particle-5")
    console.mkdir_p(output_path)

    data = joblib.load(pkl_file)

    policy = data["policy"]
    trainer = data['trainer']

    env_cls = trainer.env_cls

    for idx in range(100):
        target_seed = np.random.randint(np.iinfo(np.int32).max)
        seed = np.random.randint(np.iinfo(np.int32).max)
        demo_env = env_cls(seed=seed, target_seed=target_seed)
        analogy_env = env_cls(seed=seed + 10007, target_seed=target_seed)
        demo_policy = trainer.demo_policy_cls(demo_env)

        print("Generating %d_demo.mp4" % idx)
        encoder = ImageEncoder(output_path=osp.join(output_path, '%d_demo.mp4' % idx),
                               frame_shape=(100, 100, 3),
                               frames_per_sec=4)

        observations = []
        actions = []
        rewards = []
        obs = demo_env.reset()
        for t in range(20):
            encoder.capture_frame(to_img(obs))
            action, _ = demo_policy.get_action(obs)
            next_obs, reward, done, info = demo_env.step(action)
            observations.append(demo_env.observation_space.flatten(obs))
            actions.append(demo_env.action_space.flatten(action))
            rewards.append(reward)
            obs = next_obs
        encoder.close()


        print("Generating %d_analogy.mp4" % idx)
        encoder = ImageEncoder(output_path=osp.join(output_path, '%d_analogy.mp4' % idx),
                               frame_shape=(100, 100, 3),
                               frames_per_sec=4)

        obs = analogy_env.reset()
        policy.reset()
        policy.apply_demo(dict(observations=observations, actions=actions, rewards=rewards))
        for t in range(20):
            encoder.capture_frame(to_img(obs))
            action, _ = policy.get_action(obs)
            next_obs, reward, done, info = analogy_env.step(action)
            obs = next_obs
        encoder.close()
