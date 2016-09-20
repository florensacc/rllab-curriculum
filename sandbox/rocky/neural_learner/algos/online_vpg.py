from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
import numpy as np


class OnlineVPG(object):
    def __init__(
            self,
            env,
            policy,
            vf,
            n_envs=16,
            max_path_length=500,
            update_interval=5,
    ):
        self.env = env
        self.policy = policy
        self.vf = vf
        self.n_envs = n_envs
        self.max_path_length = max_path_length

    def train(self):
        vec_env = VecEnvExecutor(
            self.env,
            n=self.n_envs,
            max_path_length=self.max_path_length
        )

        observations = vec_env.reset()
        dones = np.asarray([True] * self.n_envs)

        while True:
            self.policy.reset(dones)
            actions, agent_infos = self.policy.get_actions(observations)
            next_obses, rewards, dones, env_infos = vec_env.step(actions)


            observations = next_obses



            sampler.obtain_samples()

        sampler.shutdown_worker()
