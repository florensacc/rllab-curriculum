import os.path

from gym.monitoring.video_recorder import ImageEncoder
# from rllab.misc.instrument import run_experiment_lite

from rllab import config
import joblib
import tensorflow as tf
import numpy as np

from rllab.misc.console import mkdir_p
from rllab.misc.ext import using_seed
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.s3.resource_manager import resource_manager
from sandbox.rocky.tf.envs.base import TfEnv

MODE = "local_docker"
# MODE = "local"

resource_name = "irl/I1-3k-bc-pretrained.pkl"


# path = "data/s3/copter-1/copter-1_2016_11_29_14_39_17_0004/params.pkl"

# with tf.Session() as sess:
#
#     data = joblib.load(path)
#
#     for seed in range(100):
#         with using_seed(seed):
#
#
#     import ipdb; ipdb.set_trace()

def run_task(*_):
    from sandbox.rocky.new_analogy.envs.conopt_env import ConoptEnv
    output_path = os.path.join(config.PROJECT_PATH, "data/video/copter-bc")
    mkdir_p(output_path)
    with tf.Session() as sess:

        path = resource_manager.get_file(resource_name)
        data = joblib.load(path)

        # env = data['env']
        policy = data['policy']

        traj_data = np.load("/shared-data/I1-3k-data.npz")

        exp_x = traj_data["exp_x"]
        exp_u = traj_data["exp_u"]
        exp_rewards = traj_data["exp_rewards"]
        exp_task_ids = traj_data["exp_task_ids"]
        # logger.log("Loaded")
        paths = []
        for xs, us, rewards, task_id in zip(exp_x, exp_u, exp_rewards[:, :, 0], exp_task_ids):
            # Filter by performance
            if rewards[-1] >= 5.3:
                paths.append(
                    dict(
                        observations=xs,
                        actions=us,
                        rewards=rewards,
                        env_infos=dict(
                            task_id=np.asarray([task_id] * len(xs))
                        )
                    )
                )

        paths = np.asarray(paths)

        path_task_ids = np.asarray([p["env_infos"]["task_id"][0] for p in paths])

        # for idx in range(100):#, (x, task_id) in enumerate(zip(exp_x, exp_task_ids)):
        #
        #     with using_seed(idx):
        #         env = TfEnv(ConoptEnv("I1", seed=idx))
        #         task_id = env.wrapped_env.conopt_env.task_id
        #         print(task_id)
        # sys.exit()

        for idx in range(100):  # , (x, task_id) in enumerate(zip(exp_x, exp_task_ids)):
            # for idx in range(100):


            with using_seed(idx):
                env = TfEnv(ConoptEnv("I1_copter_3_targets"))
                obs = env.reset()

                task_id = env.wrapped_env.conopt_env.task_id

                path = paths[np.random.choice(np.where(path_task_ids == task_id)[0])]
                # import ipdb; ipdb.set_trace()
                policy.apply_demo(path)
                policy.reset()

                done = False

                file_output_path = os.path.join(output_path, 'task_%d_demo_%04d.mp4' % (task_id, idx))
                # file_output_path = os.path.join(output_path, 'demo_%04d.mp4' % idx)

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

                dimx = env.wrapped_env.conopt_env.world.dimx

                for obs in path["observations"]:
                    try:
                        env.reset()
                        env.wrapped_env.conopt_env.reset_to(obs[:dimx])
                        joint_img = env.render(mode='rgb_array')
                        encoder.capture_frame(joint_img)
                    except Exception as e:
                        import ipdb;
                        ipdb.set_trace()

                env.reset()

                # replay demo

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
                    os.system("mv %s %s" % (file_output_path, os.path.join(output_path, 'task_%d_good_demo_%04d.mp4' % (
                        task_id, idx))))
                else:
                    os.system("mv %s %s" % (file_output_path, os.path.join(output_path, 'task_%d_bad_demo_%04d.mp4' % (
                        task_id, idx))))


if MODE == "local":
    env = dict(PYTHONPATH=":".join([
        config.PROJECT_PATH,
        os.path.join(config.PROJECT_PATH, "conopt_root"),
    ]))
else:
    env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["DISPLAY"] = ":99"


if MODE in ["local"]:
    pre_commands = []
    python_command = "python"
else:
    pre_commands = [
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa/:$LD_LIBRARY_PATH"
    ]
    python_command = 'xvfb-run -a -s "-ac -screen 0 1400x900x24 +extension RANDR" -- python'

run_experiment_lite(
    run_task,
    use_cloudpickle=True,
    # exp_prefix="-i1",
    mode=MODE,
    use_gpu=True,
    snapshot_mode="last",
    sync_all_data_node_to_s3=False,
    n_parallel=0,
    env=env,
    docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    python_command=python_command,
    pre_commands=pre_commands,
    variant=dict(),
    seed=0,
)
