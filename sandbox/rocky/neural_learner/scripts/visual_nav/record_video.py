import os.path

from gym.monitoring.video_recorder import ImageEncoder

from rllab import config
import joblib
import tensorflow as tf
import numpy as np

from rllab.misc.console import mkdir_p
from sandbox.rocky.s3.resource_manager import resource_manager

output_path = os.path.join(config.PROJECT_PATH, "data/video/doom-maze-3x3-2-episodes")

mkdir_p(output_path)
# file_path = os.path.join(config.PROJECT_PATH,
#                          "data/saved_params/doom-maze-73-1_2016_10_27_21_15_13_0016.pkl")
# file_path = os.path.join(config.PROJECT_PATH,
#                          "data/saved_params/doom-maze-77_2016_10_30_22_11_44_0004.pkl")


# resource_name = "saved_params/doom-maze/doom-maze-77_2016_10_30_22_11_44_0004.pkl"
#
# resource_manager.register_file(resource_name, file_name=file_path)



resource_name = "saved_params/doom-maze/doom-maze-77_2016_10_30_22_11_44_0012.pkl"
file_name = resource_manager.get_file(resource_name)

with tf.Session() as sess:
    data = joblib.load(file_name)

    env = data['env']
    policy = data['policy']

    for idx in range(100):
        obs = env.reset()
        policy.reset()

        done = False

        file_output_path = os.path.join(output_path, 'demo_%04d.mp4' % idx)

        encoder = ImageEncoder(
            output_path=file_output_path,
            frame_shape=(400, 800, 3),
            frames_per_sec=15  # 4
        )

        t = 0

        max_path_length = 500

        obs_list = []

        while not done and t < max_path_length:
            action, _ = policy.get_action(obs)
            joint_img = env.render(mode='rgb_array', full_size=True)
            joint_img = joint_img[:, :, ::-1]
            next_obs, _, done, _ = env.step(action)
            encoder.capture_frame(joint_img)
            obs_list.append(obs)
            obs = next_obs
            t += 1
            print(t)

        encoder.close()

        traj1_len = np.where([x[-1] for x in obs_list])[0][-1]
        traj2_len = len(obs_list) - traj1_len

        diff = traj1_len - traj2_len

        if diff > 0:
            os.system("mv %s %s" % (file_output_path, os.path.join(output_path, 'good_%03d_demo_%04d.mp4' % (abs(diff),
                                                                                                             idx))))
        else:
            os.system("mv %s %s" % (file_output_path, os.path.join(output_path, 'bad_%03d_demo_%04d.mp4' % (abs(diff),
                                                                                                            idx))))
            # import ipdb; ipdb.set_trace()
#
