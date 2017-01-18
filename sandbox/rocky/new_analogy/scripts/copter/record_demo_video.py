import os.path

from gym.monitoring.video_recorder import ImageEncoder
from rllab.misc.instrument import run_experiment_lite

from rllab import config


MODE = "local_docker"


def run_task(*_):
    file = "/shared-data/copter-5k-data.npz"
    import numpy as np
    from rllab.misc.ext import using_seed
    from conopt import envs

    data = np.load(file)

    exp_x = data["exp_x"]
    exp_task_ids = data["exp_task_ids"]

    with using_seed(0):
        experiment = envs.load("I1")
        env = experiment.make()

    output_path = os.path.join(config.PROJECT_PATH, "data/video/copter-demo")

    from rllab.misc.console import mkdir_p
    mkdir_p(output_path)

    for idx, (x, task_id) in enumerate(zip(exp_x, exp_task_ids)):
        print(idx)

        file_output_path = os.path.join(output_path, 'task_%d_demo_%04d.mp4' % (task_id, idx))

        encoder = ImageEncoder(
            output_path=file_output_path,
            frame_shape=(1000, 1000, 3),
            frames_per_sec=15
        )

        for i in range(x.shape[0]):
            env.reset_to(x[i][:env.world.dimx])
            joint_img = env.render(mode='rgb_array')
            encoder.capture_frame(joint_img)

        encoder.close()


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

pre_commands = [
    "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa/:$LD_LIBRARY_PATH"
]

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
    python_command='xvfb-run -a -s "-ac -screen 0 1400x900x24 +extension RANDR" -- python',
    pre_commands=pre_commands,
    variant=dict(),
    seed=0,
)
