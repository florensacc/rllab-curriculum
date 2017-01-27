from collections import deque

import itertools
import pickle
import numpy as np

from rllab.misc import tensor_utils
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor


class PathSegsRecorder(object):
    def __init__(self, env_spec, n):
        self.env_spec = env_spec
        self.n = n
        self.complete_paths = []
        self.running_paths = [None] * n
        self.n_samples = 0
        self.n_complete_samples = 0
        self.path_rewards = [0] * n
        self.complete_path_rewards = []
        self.sliding_complete_path_rewards = deque(maxlen=100)

    def record(self, obses, actions, next_obses, rewards, dones, agent_infos, env_infos):
        agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
        env_infos = tensor_utils.split_tensor_dict_list(env_infos)
        if env_infos is None:
            env_infos = [dict() for _ in range(self.n)]
        if agent_infos is None:
            agent_infos = [dict() for _ in range(self.n)]
        for idx, observation, action, next_obs, reward, env_info, agent_info, done in zip(
                itertools.count(), obses, actions, next_obses, rewards, env_infos, agent_infos, dones):
            if self.running_paths[idx] is None:
                self.running_paths[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                    dones=[],
                    start_t=0,
                )
            path = self.running_paths[idx]
            path["observations"].append(observation)
            path["actions"].append(action)
            path["rewards"].append(reward)
            path["env_infos"].append(env_info)
            path["agent_infos"].append(agent_info)
            path["last_observation"] = next_obs
            path["dones"].append(done)
            self.path_rewards[idx] += reward
            self.n_samples += 1
            if done:
                self.complete_paths.append(self.flatten_path(path))
                self.complete_path_rewards.append(self.path_rewards[idx])
                self.n_complete_samples += len(path["rewards"])
                self.running_paths[idx] = None
                self.path_rewards[idx] = 0

    def flatten_path(self, raw_path):
        return dict(
            observations=self.env_spec.observation_space.flatten_n(raw_path["observations"]),
            actions=self.env_spec.action_space.flatten_n(raw_path["actions"]),
            rewards=tensor_utils.stack_tensor_list(raw_path["rewards"]),
            env_infos=tensor_utils.stack_tensor_dict_list(raw_path["env_infos"]),
            agent_infos=tensor_utils.stack_tensor_dict_list(raw_path["agent_infos"]),
            dones=tensor_utils.stack_tensor_list(raw_path["dones"]),
            last_observation=self.env_spec.observation_space.flatten(raw_path["last_observation"]),
            start_t=raw_path["start_t"],
        )

    def dump_path_segs(self):
        path_segs = list(self.complete_paths)
        self.complete_paths = []
        for seg in self.running_paths:
            if seg is not None:
                path_segs.append(self.flatten_path(seg))
        for idx, path in enumerate(self.running_paths):
            if path is not None:
                # have the proper start index in the next iteration
                self.running_paths[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                    dones=[],
                    start_t=len(path["rewards"]),
                )
        self.n_samples = 0
        self.n_complete_samples = 0
        logger.record_tabular_misc_stat('Return', self.complete_path_rewards, placement='front')
        self.sliding_complete_path_rewards.extend(self.complete_path_rewards)
        logger.record_tabular_misc_stat('SlidingReturn', np.asarray(self.sliding_complete_path_rewards),
                                        placement='front')
        self.complete_path_rewards = []
        return path_segs


def sample_rollouts(env, policy, batch_size, max_path_length, n_envs, parallel):
    if getattr(env, 'vectorized', False):
        vec_env = env.vec_env_executor(n_envs=n_envs)
    elif parallel:
        vec_env = ParallelVecEnvExecutor(
            env=env,
            n_envs=n_envs,
        )
    else:
        envs = [pickle.loads(pickle.dumps(env)) for _ in range(n_envs)]
        vec_env = VecEnvExecutor(
            envs=envs,
        )
    recorder = PathSegsRecorder(env_spec=env.spec, n=n_envs)
    try:
        dones = np.asarray([True] * n_envs)
        obses = vec_env.reset(dones)
        policy.reset(dones)
        while True:
            actions, agent_infos = policy.get_actions(obses)
            next_obses, rewards, dones, env_infos = vec_env.step(actions, max_path_length=max_path_length)
            recorder.record(
                obses=obses,
                actions=actions,
                next_obses=next_obses,
                rewards=rewards,
                dones=dones,
                agent_infos=agent_infos,
                env_infos=env_infos
            )
            obses = next_obses
            if np.any(dones):
                reset_obs = vec_env.reset(dones)
                done_cnt = 0
                for idx, done in enumerate(dones):
                    if done:
                        obses[idx] = reset_obs[done_cnt]
                        done_cnt += 1
                policy.reset(dones)
            if recorder.n_samples >= batch_size:
                yield recorder.dump_path_segs()
    finally:
        vec_env.terminate()

