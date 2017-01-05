import os.path

from gym.monitoring.video_recorder import ImageEncoder

from rllab import config
import joblib
import tensorflow as tf

from rllab.misc.console import mkdir_p
from rllab.misc.ext import using_seed
from sandbox.rocky.neural_learner.scripts.visual_nav.utils import update_exp_pkl

# # trpo
# folder_name = "data/s3/gail-claw-5/gail-claw-5_2016_11_13_19_28_07_0004"
# output_path = os.path.join(config.PROJECT_PATH, "data/video/trpo-claw")
# mkdir_p(output_path)

# # gail
# folder_name = "data/s3/gail-claw-5/gail-claw-5_2016_11_13_19_38_04_0004"
# output_path = os.path.join(config.PROJECT_PATH, "data/video/gail-claw")
# mkdir_p(output_path)

# # trpo finetune
# folder_name = "data/s3/trpo-finetune-claw-1/trpo-finetune-claw-1_2016_11_14_11_25_04_0008"
# output_path = os.path.join(config.PROJECT_PATH, "data/video/trpo-finetune-claw")
# mkdir_p(output_path)

# bc
# folder_name = "data/s3/bc-claw-2/bc-claw-2_2016_11_13_21_16_08_0020"
# output_path = os.path.join(config.PROJECT_PATH, "data/video/bc-claw")
# mkdir_p(output_path)

# # gail finetune
# folder_name = "data/s3/gail-finetune-claw-2/gail-finetune-claw-2_2016_11_15_14_51_12_0014"
# output_path = os.path.join(config.PROJECT_PATH, "data/video/gail-finetune-claw")
# mkdir_p(output_path)

# gcl finetune
folder_name = "data/s3/gcl-finetune-claw-1/gcl-finetune-claw-1_2016_11_15_17_21_04_0024"
output_path = os.path.join(config.PROJECT_PATH, "data/video/gcl-finetune-claw-good-pol")
mkdir_p(output_path)

exp_prefix = folder_name.split("/")[2]
exp_name = folder_name.split("/")[3]

pkl_file = os.path.join(folder_name, "params.pkl")

if not os.path.exists(pkl_file):
    update_exp_pkl(exp_prefix=exp_prefix, exp_name=exp_name)

with tf.Session() as sess:
    data = joblib.load(pkl_file)

    env = data['env']
    policy = data['policy']

    for idx in range(100):

        with using_seed(idx):
            obs = env.reset()
            policy.reset()

            done = False

            file_output_path = os.path.join(output_path, 'demo_%04d.mp4' % idx)

            encoder = ImageEncoder(
                output_path=file_output_path,
                frame_shape=(1000, 1000, 3),
                frames_per_sec=15  # 4
            )

            t = 0

            max_path_length = 100

            obs_list = []

            rewards = []

            reward = 0

            while not done and t < max_path_length:
                action, _ = policy.get_action(obs)
                joint_img = env.render(mode='rgb_array')
                next_obs, reward, done, _ = env.step(action)
                encoder.capture_frame(joint_img)
                obs_list.append(obs)
                obs = next_obs
                t += 1
                print(t)

            encoder.close()

            if reward >= 4:
                os.system("mv %s %s" % (file_output_path, os.path.join(output_path, 'good_demo_%04d.mp4' % (idx))))
            else:
                os.system("mv %s %s" % (file_output_path, os.path.join(output_path, 'bad_demo_%04d.mp4' % (idx))))
