import multiprocessing

from rllab import config
import os
import joblib
import tensorflow as tf
import numpy as np

from rllab.envs.base import Env
from rllab.envs.proxy_env import ProxyEnv
import rllab.envs.proxy_env
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
#     10: 'random-mdp-6_2016_10_21_19_02_28_0003',
#     # 50: 'random-mdp-9_2016_10_22_11_00_39_0015'
# }

exps = {
    25: 'random-mdp-12_2016_10_28_17_32_40_0005',
    10: 'random-mdp-12_2016_10_27_01_21_11_0002',
    75: 'random-mdp-12_2016_10_28_17_32_40_0010',
    100: 'random-mdp-12_2016_10_27_01_21_11_0012',
    50: 'random-mdp-12_2016_10_27_01_21_11_0009'
}


class OldProxyEnv(Env):
    def __init__(self, wrapped_env):
        # Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self):
        return self._wrapped_env.reset()

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)


old_proxy_env = OldProxyEnv
proxy_env = ProxyEnv

if __name__ == "__main__":
    # file_path = os.path.join(config.PROJECT_PATH, "data/s3/mab-7/mab-7_2016_10_17_00_12_02_0134/params.pkl")

    for (n_episodes), exp_name in exps.items():
        folder = exp_name.split("_")[0]
        file_path = os.path.join(config.PROJECT_PATH, "data/s3/{folder}/{exp_name}/params.pkl".format(
            folder=folder,
            exp_name=exp_name

        ))
        print(file_path)


        def run():
            tf.reset_default_graph()
            with tf.Session() as sess:
                data = joblib.load(file_path)

                policy = data['policy']
                env = data['env']

                n_trials = 1000
                # env.wrapped_env.n_episodes = 50  # 100#0

                n_episodes = env.wrapped_env.n_episodes  # 1000

                # import ipdb; ipdb.set_trace()

                print(n_episodes)
                #
                # env.wrapped_env.n_episodes = n_episodes

                sampler = VectorizedSampler(env, policy, n_envs=n_trials)
                sampler.start_worker()

                episode_horizon = 10

                paths = sampler.obtain_samples(0, max_path_length=n_episodes * episode_horizon,
                                               batch_size=n_trials * n_episodes * episode_horizon)

                returns = [np.sum(p["rewards"]) for p in paths]

                # import ipdb; ipdb.set_trace()

                print(n_episodes, np.mean(returns), np.std(returns) / np.sqrt(len(returns) - 1))


        try:
            rllab.envs.proxy_env.ProxyEnv = proxy_env
            # rllab.envs.proxy_env.ProxyEnv = old_proxy_env
            run()
        except Exception as e:
            run()
