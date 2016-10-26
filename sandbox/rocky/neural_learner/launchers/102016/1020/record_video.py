import os.path

from gym.monitoring.video_recorder import ImageEncoder

from rllab import config
import joblib
import tensorflow as tf

output_path = os.path.join(config.PROJECT_PATH, "data/video/doom-two-goal-maze")
file_path = os.path.join(config.PROJECT_PATH, "data/saved_params/doom-maze-35-5_2016_10_20_13_34_18_0003.pkl")


with tf.Session() as sess:
    data = joblib.load(file_path)

    env = data['env']
    policy = data['policy']


    for idx in range(100):
        obs = env.reset()
        policy.reset()

        done = False

        encoder = ImageEncoder(
            output_path=os.path.join(output_path, 'demo_%04d.mp4' % idx),
            frame_shape=(400, 800, 3),
            frames_per_sec=15#4
        )

        t = 0

        while not done and t < 300:
            action, _ = policy.get_action(obs)
            joint_img = env.render(mode='rgb_array', full_size=True)
            joint_img = joint_img[:, :, ::-1]
            next_obs, _, done, _ = env.step(action)
            # import ipdb; ipdb.set_trace()
            encoder.capture_frame(joint_img)
            obs = next_obs
            t += 1
            print(t)

        encoder.close()




        # import ipdb; ipdb.set_trace()
    #
