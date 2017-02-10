import pickle
import tempfile

import joblib
import numpy as np

from rllab.misc.instrument import run_experiment_lite
from rllab.sampler import parallel_sampler
from sandbox.rocky.new_analogy.tf.policies.deterministic_policy import DeterministicPolicy
from sandbox.rocky.s3.resource_manager import resource_manager
from sandbox.rocky.tf.envs.base import TfEnv


def gen_data(*_):
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from gpr_package.bin import tower_fetch_policy as tower
    np.random.seed(0)
    task_id = tower.get_task_from_text("ab")
    import tensorflow as tf

    with tf.Session() as sess:
        horizon = 1000

        env = TfEnv(GprEnv("fetch.sim_fetch", task_id=task_id, experiment_args=dict(nboxes=2, horizon=horizon)))
        file_name = resource_manager.get_file("tower_fetch_ab_pretrained_round_1")
        with open(file_name, "rb") as f:
            data = joblib.load(file_name)
            policy = DeterministicPolicy(env_spec=env.spec, wrapped_policy=data["policy"])

        n_trajs = 10000

        parallel_sampler.populate_task(env, policy)
        paths = parallel_sampler.sample_paths(policy.get_param_values(), max_samples=n_trajs * horizon,
                                       max_path_length=horizon)

        # sampler = VectorizedSampler(env=env, policy=policy, n_envs=100)
        # sampler.start_worker()
        # paths = sampler.obtain_samples(itr=0, max_path_length=horizon, batch_size=n_trajs * horizon,
        #                                max_n_trajs=n_trajs)

        print("Success rate: ", np.mean(np.asarray([p["rewards"][-1] for p in paths]) > 4))
        f_name = tempfile.NamedTemporaryFile().name + ".pkl"
        with open(f_name, "wb") as f:
            pickle.dump(paths, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        resource_manager.register_file("tower_fetch_ab_on_policy_round_1", f_name)
        # sampler.shutdown_worker()


run_experiment_lite(
    gen_data,
    use_cloudpickle=True,
    mode="local_docker",
    n_parallel=32,
    seed=0,
    env=dict(CUDA_VISIBLE_DEVICES="0", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
    # docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170114",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
)
