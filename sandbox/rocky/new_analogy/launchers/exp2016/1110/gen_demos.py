import tensorflow as tf

import cloudpickle

from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.s3.resource_manager import resource_manager, tmp_file_name
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from rllab.misc import logger


class VG(VariantGenerator):
    @variant
    def task(self):
        return ["half_cheetah", "ant", "swimmer", "walker2d", "hopper"]

    @variant
    def deterministic(self):
        return [True]#, False]


MODE = launch_cirrascale("pascal")


def run_task(vv):
    with tf.Session() as sess:
        task = vv["task"]
        resource_name = "pretrained_models/{task}.pkl".format(task=task)
        local_file = resource_manager.get_file(resource_name)
        with open(local_file, "rb") as f:
            data = cloudpickle.load(f)
        policy = data["policy"]
        env = data["env"]

        vec_sampler = VectorizedSampler(env=env, policy=policy, n_envs=100, parallel=True)
        vec_sampler.start_worker()

        deterministic = vv["deterministic"]

        if deterministic:
            sess.run(tf.assign(policy._l_std_param.param, [-9] * env.action_space.flat_dim))

        n_trajs = 1000
        max_path_length = 500

        paths = vec_sampler.obtain_samples(itr=0, max_path_length=max_path_length,
                                           batch_size=n_trajs * max_path_length,
                                           max_n_trajs=n_trajs)

        returns = [np.sum(p["rewards"]) for p in paths]

        file_name = tmp_file_name(file_ext="pkl")
        resource_name = "demo_trajs/{task}_n_trajs_{n_trajs}_horizon_{horizon}_deterministic_{" \
                        "deterministic}.pkl".format(task=task, n_trajs=str(n_trajs), horizon=str(
                            max_path_length), deterministic=str(deterministic))

        with open(file_name, "wb") as f:
            cloudpickle.dump(paths, f, protocol=3)

        resource_manager.register_file(resource_name, file_name=file_name)

        logger.record_tabular_misc_stat('Return', returns, placement='front')
        logger.dump_tabular()


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="trpo-gen-demos-1",
        variant=v,
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=8,
        docker_image="dementrock/rllab3-shared-gpu-cuda80",
    )
