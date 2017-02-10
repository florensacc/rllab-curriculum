import copy
import inspect
import pickle

from rllab.sampler.base import Sampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class VectorizedSampler(Sampler):

    def __init__(self, env, policy, n_envs, vec_env=None, parallel=False):
        self.env = env
        self.policy = policy
        self.n_envs = n_envs
        self.vec_env = vec_env
        self.env_spec = env.spec
        self.parallel = parallel

    def start_worker(self):
        if self.vec_env is None:
            n_envs = self.n_envs
            if getattr(self.env, 'vectorized', False):
                self.vec_env = self.env.vec_env_executor(n_envs=n_envs)
            elif self.parallel:
                self.vec_env = ParallelVecEnvExecutor(
                    env=self.env,
                    n_envs=self.n_envs,
                )
            else:
                envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]
                self.vec_env = VecEnvExecutor(
                    envs=envs,
                )

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, max_path_length, batch_size, max_n_trajs=None, seeds=None, show_progress=True):
        paths = []
        n_samples = 0
        dones = np.asarray([True] * self.vec_env.n_envs)
        obses = self.vec_env.reset(dones, seeds=seeds)

        running_paths = [None] * self.vec_env.n_envs

        if show_progress:
            pbar = ProgBarCounter(batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.policy
        if hasattr(policy, 'inform_vec_env'):
            policy.inform_vec_env(self.vec_env)

        import time
        while n_samples < batch_size:
            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, max_path_length=max_path_length)

            if np.any(dones):
                new_obses = self.vec_env.reset(dones)
                reset_idx = 0
                for idx, done in enumerate(dones):
                    if done:
                        next_obses[idx] = new_obses[reset_idx]
                        reset_idx += 1

            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.n_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.n_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None

            if max_n_trajs is not None and len(paths) >= max_n_trajs:
                break

            process_time += time.time() - t
            if show_progress:
                pbar.inc(len(obses))
            obses = next_obses

        if show_progress:
            pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths
