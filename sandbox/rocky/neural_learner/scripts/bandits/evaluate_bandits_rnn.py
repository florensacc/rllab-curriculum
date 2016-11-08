import multiprocessing

from rllab import config
import os
import joblib
import tensorflow as tf
import numpy as np
from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.cirrascale.launch_job import get_directory
import subprocess


def check_host(host, folder):
    try:
        command = [
            "ssh",
            "rocky@" + host,
            "find /local_home/rocky/rllab-workdir -name %s" % folder,
        ]
        # print(" ".join(command))
        file_location = subprocess.check_output(command).decode()
        if len(file_location) > 0:
            return (host, file_location)
    except Exception as e:
        pass
        # print(e)

    return None


# exps = {
#     (5, 500): 'mab-10-1_2016_10_20_13_40_15_0074',
#     (5, 10): 'mab-1_2016_10_13_23_20_05_0035',
#     (10, 500): 'mab-7_2016_10_17_00_12_02_0096',
#     (10, 10): 'mab-1_2016_10_13_23_20_05_0022',
#     (50, 100): 'mab-1_2016_10_13_23_20_05_0005',
#     (5, 100): 'mab-1_2016_10_13_23_20_05_0028',
#     (10, 100): 'mab-1_2016_10_13_23_20_05_0017',
#     (50, 500): 'mab-7_2016_10_17_00_12_02_0196',
#     (50, 10): 'mab-1_2016_10_13_23_20_05_0012'
# }

exps = {
    (5, 10): 'mab-17_2016_10_30_17_08_35_0031',
    (5, 100): 'mab-17_2016_10_30_17_08_35_0039',
    (5, 500): 'mab-17_2016_10_30_17_08_35_0041',
    (10, 10): 'mab-17_2016_10_30_17_08_35_0019',
    (10, 100): 'mab-17_2016_10_30_17_08_35_0025',
    (10, 500): 'mab-17_2016_10_30_17_08_35_0028',
    (50, 10): 'mab-17_2016_10_30_17_08_35_0003',
    (50, 100): 'mab-17_2016_10_30_17_08_35_0007',
    (50, 500): 'mab-17_2016_10_30_17_08_35_0013',
}

if __name__ == "__main__":
    # file_path = os.path.join(config.PROJECT_PATH, "data/s3/mab-7/mab-7_2016_10_17_00_12_02_0134/params.pkl")

    for (n_arms, n_episodes), exp_name in exps.items():
        folder = exp_name.split("_")[0]
        file_path = os.path.join(config.PROJECT_PATH, "data/s3/{folder}/{exp_name}/params.pkl".format(
            folder=folder,
            exp_name=exp_name

        ))

        # dir = get_directory()
        #
        # with multiprocessing.Pool(100) as pool:
        #     results = pool.starmap(check_host, [(host, job_name) for host in dir.keys()])
        #
        # for result in results:
        #     if result is not None:
        #         host, paths = result
        #         paths = paths.split("\n")
        #         for path in paths:
        #             if len(path) > 0 and "data/local" in path:
        #                 command = [
        #                     "scp",
        #                     "rocky@{host}:{path}/params.pkl".format(host=host, path=path),
        #                     "data/s3/mab-10-1/{job_name}/params.pkl".format(job_name=job_name)
        #                 ]
        #                 print(" ".join(command))
        #                 subprocess.check_call(command)

        # for host in dir.keys():
        #     check_host(host, job_name)


        # import ipdb; ipdb.set_trace()

        tf.reset_default_graph()

        with tf.Session() as sess:
            data = joblib.load(file_path)
            policy = data['policy']
            env = data['env']

            n_trials = 1000
            n_episodes = env.wrapped_env.n_episodes  # 1000
            #
            # env.wrapped_env.n_episodes = n_episodes

            sampler = VectorizedSampler(env, policy, n_envs=n_trials)
            sampler.start_worker()

            paths = sampler.obtain_samples(0, max_path_length=n_episodes, batch_size=n_trials * n_episodes)

            returns = [np.sum(p["rewards"]) for p in paths]

            print(n_arms, n_episodes, np.mean(returns), np.std(returns) / np.sqrt(len(returns) - 1))
