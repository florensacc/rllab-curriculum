import pickle
from collections import defaultdict

from rllab.core.serializable import Serializable
from sandbox.rocky.chainer.envs.vec_env_executor import VecEnvExecutor
import numpy as np
import chainer.functions as F
import chainer

from sandbox.rocky.chainer.misc import tensor_utils
from rllab.misc import logger
from rllab.misc import special
from rllab.misc import ext
import multiprocessing as mp

from sandbox.rocky.chainer.optimizers.rmsprop_async import RMSpropAsync
from sandbox.rocky.chainer.policies.base import StochasticPolicy
import queue


class SharedPolicy(StochasticPolicy, chainer.Link):  # , Serializable):
    def __init__(self, wrapped_policy, shared_params=None):
        if shared_params is None:
            shared_params = []
            for param in wrapped_policy.get_params():
                shared_params.append(
                    np.frombuffer(
                        mp.RawArray('f', param.data.flatten()),
                        dtype=param.data.dtype
                    ).reshape(param.data.shape)
                )
        for idx, param in enumerate(wrapped_policy.get_params()):
            param.data = shared_params[idx]
        # Serializable.quick_init(self, locals())
        StochasticPolicy.__init__(self, wrapped_policy.env_spec)
        chainer.Link.__init__(self)
        self.wrapped_policy = wrapped_policy

    def get_actions(self, observations):
        return self.wrapped_policy.get_actions(observations)

    def get_actions_sym(self, observations):
        return self.wrapped_policy.get_actions_sym(observations)

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params(**tags)

    def dist_info_sym(self, obs_var, state_info_vars):
        return self.wrapped_policy.dist_info_sym(obs_var, state_info_vars)

    @property
    def distribution(self):
        return self.wrapped_policy.distribution

    def namedparams(self):
        yield from self.wrapped_policy.namedparams()

    def params(self):
        yield from self.wrapped_policy.params()

    def log_diagnostics(self, paths):
        self.wrapped_policy.log_diagnostics(paths)


class SharedOptimizer(Serializable):
    def __init__(self, wrapped_optimizer, shared_params=None):
        if shared_params is None:
            shared_params = []
            for _, param_dict in sorted(wrapped_optimizer._states.items()):
                for k, param in list(sorted(param_dict.items())):
                    shared_params.append(np.frombuffer(
                        mp.RawArray('f', param.flatten()),
                        dtype=param.dtype
                    ).reshape(param.shape))
        idx = 0
        for _, param_dict in sorted(wrapped_optimizer._states.items()):
            for k, param in list(sorted(param_dict.items())):
                param_dict[k] = shared_params[idx]
                idx += 1
        Serializable.quick_init(self, locals())
        self.wrapped_optimizer = wrapped_optimizer

    def update(self):
        self.wrapped_optimizer.update()


class A3C(object):
    """
    Start working on an A3C implementation integrated with rllab's interface
    """

    def __init__(
            self,
            env,
            policy,
            # shared_policy,
            optimizer,
            shared_params,
            shared_states,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1.,
            n_envs_per_worker=1,
            update_frequency=5,
            # learning_rate=1e-3,
            epoch_length=1000,
            n_epochs=1000,
            n_parallel=None,
            normalize_momentum=0.9,
            scale_reward=1.0,
            # share_optimizer=False,
            normalize_adv=True,
            normalize_vf=True,
            max_grad_norm=40.0,
            # optimizer_type="rmsprop_async",
            policy_loss_coeff=1.,
            vf_loss_coeff=0.5,
            entropy_bonus_coeff=1e-2,
    ):
        self.env = env
        self.policy = policy
        # self.shared_policy = shared_policy#SharedPolicy(pickle.loads(pickle.dumps(policy)))
        # sync parameters
        # self.policy.set_param_values_from(self.shared_policy)
        self.n_envs_per_worker = n_envs_per_worker
        self.update_frequency = update_frequency
        # self.learning_rate = learning_rate
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.shared_params=shared_params
        self.shared_states=shared_states
        # if max_grad_norm is not None:
        #     optimizer.add_hook(chainer.optimizer.GradientClipping(max_grad_norm))
        # if share_optimizer:
        #     self.optimizer = SharedOptimizer(optimizer)
        # else:
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.normalize_momentum = normalize_momentum
        self.running_adv_mean = mp.Value('f', 0.)
        self.running_adv_std = mp.Value('f', 1.)
        self.running_vf_mean = mp.Value('f', 0.)
        self.running_vf_std = mp.Value('f', 1.)
        self.normalize_adv = normalize_adv
        self.normalize_vf = normalize_vf
        self.scale_reward = scale_reward
        self.policy_loss_coeff = policy_loss_coeff
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.max_grad_norm = max_grad_norm
        if n_parallel is None:
            n_parallel = mp.cpu_count()
        self.n_parallel = n_parallel
        self.global_t = mp.Value('l', 0)
        self.loggings = defaultdict(list)

    def update_once(
            self,
            worker_id,
            seg_obs,
            seg_actions,
            seg_rewards,
            seg_dones,
            seg_env_infos,
            seg_agent_infos,
            next_agent_info,
    ):
        ##############################
        # Process data needed for optimization
        ##############################

        seg_rewards = np.asarray(seg_rewards) * self.scale_reward
        seg_dones = np.asarray(seg_dones)
        seg_vf_preds = np.asarray(
            [x['vf'].data.flatten() for x in seg_agent_infos + [next_agent_info]]
        )  # * self.running_vf_std.value + self.running_vf_mean.value

        # TODO debug
        # seg_vf_preds[...] = 0

        # seg_deltas = seg_rewards + (1 - seg_dones) * self.discount * seg_vf_preds[1:] - \
        #              seg_vf_preds[:-1]
        # seg_advs = np.empty_like(seg_rewards)
        # for t in reversed(range(len(seg_obs))):
        #     if t == len(seg_obs) - 1:
        #         seg_advs[t] = seg_deltas[t]
        #     else:
        #         seg_advs[t] = seg_deltas[t] + self.discount * self.gae_lambda * (1 - seg_dones[t]) * \
        #                                       seg_advs[t + 1]

        seg_returns = np.empty_like(seg_rewards)
        for t in reversed(range(len(seg_obs))):
            if t == len(seg_obs) - 1:
                seg_returns[t] = seg_vf_preds[t + 1]
            else:
                seg_returns[t] = seg_rewards[t] + self.discount * self.gae_lambda * (1 - seg_dones[t]) * \
                                                  seg_returns[t + 1]

        seg_advs = seg_returns - seg_vf_preds[:-1]

        # import ipdb; ipdb.set_trace()

        seg_actions = F.concat(seg_actions, axis=0)
        seg_dist_infos = dict()
        for k in self.policy.distribution.dist_info_keys:
            infos = [info[k] for info in seg_agent_infos]
            seg_dist_infos[k] = F.concat(infos, axis=0)

        seg_vf_vars = F.reshape(F.concat([info['vf'] for info in seg_agent_infos], axis=0), (-1,))

        ##############################
        # Compute objective and apply gradients
        ##############################

        # TODO this uses the GAE estimate, and the behavior is different when lambda!=1. Does it make
        # a difference?
        # vf_target = seg_advs + seg_vf_preds[:-1]
        N = seg_actions.shape[0]

        # if self.normalize_adv:
        #     # with self.running_adv_mean.get_lock():
        #     self.running_adv_mean.value = self.normalize_momentum * self.running_adv_mean.value + \
        #                                   (1 - self.normalize_momentum) * np.mean(seg_advs)
        #     self.running_adv_std.value = self.normalize_momentum * self.running_adv_std.value + \
        #                                  (1 - self.normalize_momentum) * np.std(seg_advs)
        # if self.normalize_vf:
        #     # with self.running_vf_mean.get_lock():
        #     self.running_vf_mean.value = self.normalize_momentum * self.running_vf_mean.value + \
        #                                  (1 - self.normalize_momentum) * np.mean(vf_target)
        #     self.running_vf_std.value = self.normalize_momentum * self.running_vf_std.value + \
        #                                 (1 - self.normalize_momentum) * np.std(vf_target)

        # unwhiten under new distribution
        # seg_vf_vars = seg_vf_vars * self.running_vf_std.value + self.running_vf_mean.value

        # adv_divisor = max(1., self.running_adv_std.value + 1e-8)
        # vf_divisor = max(1., self.running_vf_std.value + 1e-8)

        logli = self.policy.distribution.log_likelihood_sym(seg_actions, seg_dist_infos)

        # import ipdb; ipdb.set_trace()
        # seg_advs = (seg_advs - self.running_adv_mean.value) / adv_divisor
        policy_loss = -F.sum(logli * seg_advs.flatten()) / N

        vf_loss = F.sum(F.square(seg_vf_vars - seg_returns.flatten())) / N

        entropy_loss = -F.sum(self.policy.distribution.entropy_sym(seg_dist_infos)) / N
        joint_loss = self.policy_loss_coeff * policy_loss + self.vf_loss_coeff * vf_loss + self.entropy_bonus_coeff * \
                                                                                           entropy_loss

        # joint_loss = self.policy_loss_coeff * policy_loss + self.entropy_bonus_coeff * entropy_loss

        # if worker_id < 10:#in [0, 1, 2, 3]:
        # with self.global_t.get_lock():

        self.policy.zerograds()
        joint_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_policy.zerograds()
        from sandbox.rocky.chainer.algos_ref import copy_param
        copy_param.copy_grad(
            target_link=self.shared_policy, source_link=self.policy)
        # Update the globally shared model
        # if self.process_idx == 0:
        #     norm = self.optimizer.compute_grads_norm()
        #     logger.debug('grad norm:%s', norm)
        self.optimizer.update()

        copy_param.copy_param(target_link=self.policy,
                              source_link=self.shared_policy)
        # if self.process_idx == 0:
        #     logger.debug('update')

        # self.sync_parameters()
        # self.policy.cleargrads()
        # self.shared_policy.cleargrads()
        #
        # # entropy_loss.backward()
        #
        # joint_loss.backward()
        #
        # # for learning_rate in [1e-3, 1e-4, 1e-5]:
        # #
        # #     params_before = params = self.shared_policy.get_param_values()
        # #     info_before = self.policy.dist_info_sym(F.concat(np.asarray(seg_obs), axis=0), dict())
        # #
        # #     grads = self.policy.get_grad_values()
        # #     ms = 0.01 * grads * grads
        # #     params = params - learning_rate * grads / np.sqrt(ms + 1e-8)
        # #
        # #     self.shared_policy.set_param_values(params)
        # #     self.policy.set_param_values_from(self.shared_policy)
        # #
        # #     info_after = self.policy.dist_info_sym(F.concat(np.asarray(seg_obs), axis=0), dict())
        # #
        # #     kl = np.mean(self.policy.distribution.kl_sym(info_before, info_after).data)
        # #     print(learning_rate, kl)
        # #     self.shared_policy.set_param_values(params_before)
        # #     self.policy.set_param_values(params_before)
        #
        #
        # # manually perform rmsprop
        #
        #
        # # print([(k, np.linalg.norm(v.grad)) for k, v in self.policy.namedparams()])
        # # info_before = self.policy.dist_info_sym(F.concat(np.asarray(seg_obs), axis=0), dict())
        # # param_diff_before = np.linalg.norm(self.policy.get_param_values() - self.shared_policy.get_param_values())
        # # print(param_diff_before)
        # # param_before = self.shared_policy.get_param_values()
        # self.shared_policy.set_grad_values_from(self.policy)
        # self.optimizer.update()
        # self.policy.set_param_values_from(self.shared_policy)
        # param_after = self.shared_policy.get_param_values()
        # self.policy.set_param_values_from(self.shared_policy)
        # info_after = self.policy.dist_info_sym(F.concat(np.asarray(seg_obs), axis=0), dict())
        # kl = np.mean(self.policy.distribution.kl_sym(info_before, info_after).data)
        # print(kl)
        # import ipdb; ipdb.set_trace()
        # self.loggings["DeltaParamNorm"].append(np.linalg.norm(param_before - param_after))

        self.loggings["PolicyLoss"].append(policy_loss.data)
        self.loggings["VfLoss"].append(vf_loss.data)
        self.loggings["EntropyLoss"].append(entropy_loss.data)
        self.loggings["JointLoss"].append(joint_loss.data)
        self.loggings["AbsVfPred"].append(np.mean(np.abs(seg_vf_vars.data)))
        self.loggings["AbsVfTarget"].append(np.mean(np.abs(seg_returns)))
        self.loggings["MaxVfTarget"].append(np.max(np.abs(seg_returns)))
        self.loggings["MinVfTarget"].append(np.min(np.abs(seg_returns)))
        self.loggings["RunningAdvMean"].append(self.running_adv_mean.value)
        self.loggings["RunningAdvStd"].append(self.running_adv_std.value)
        self.loggings["RunningVfMean"].append(self.running_vf_mean.value)
        self.loggings["RunningVfStd"].append(self.running_vf_std.value)

        # if np.max(np.abs(vf_target)) > 100:
        #     import ipdb; ipdb.set_trace()
        # print({k: np.mean(v) for k, v in self.loggings.items()})#{k: v})

    def worker_train(self, worker_id, worker_seed, barrier):
        ext.set_seed(worker_seed)

        n_envs = self.n_envs_per_worker
        if getattr(self.env, 'vectorized', False):
            vec_env = self.env.vec_env_executor(n_envs=n_envs)
        else:
            envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]
            vec_env = VecEnvExecutor(
                envs=envs,
            )

        self.shared_policy = self.policy
        self.policy = pickle.loads(pickle.dumps(self.shared_policy))
        from sandbox.rocky.chainer.algos_ref import async
        async.set_shared_params(self.shared_policy, self.shared_params)
        async.set_shared_states(self.optimizer, self.shared_states)

        self.policy.set_param_values_from(self.shared_policy)

        obs = vec_env.reset(dones=[True] * n_envs)
        self.policy.reset(dones=[True] * n_envs)

        # if (barrier.wait()) == 0:
        #     # run normalization code
        #     self.policy.apply_normalization(obs)
        #     self.shared_policy.set_param_values_from(self.policy)
        #
        # barrier.wait()

        # if worker_id == 0:
        # normalize

        local_t = 0
        # global_t = 0
        last_log_t = 0

        seg_obs = []
        seg_actions = []
        seg_rewards = []
        seg_dones = []
        seg_env_infos = []
        seg_agent_infos = []

        paths = []
        running_paths = [None] * n_envs

        while self.global_t.value < self.n_epochs * self.epoch_length:

            local_t += n_envs
            with self.global_t.get_lock():
                self.global_t.value += n_envs
            actions_var, agent_info_vars = self.policy.get_actions_sym(obs)
            actions = actions_var.data

            next_obs, rewards, dones, env_infos = vec_env.step(
                action_n=actions, max_path_length=self.max_path_length)

            if len(seg_obs) >= self.update_frequency:

                # if worker_id > 0:
                self.update_once(
                    worker_id=worker_id,
                    seg_obs=seg_obs,
                    seg_actions=seg_actions,
                    seg_rewards=seg_rewards,
                    seg_dones=seg_dones,
                    seg_env_infos=seg_env_infos,
                    seg_agent_infos=seg_agent_infos,
                    next_agent_info=agent_info_vars,
                )

                ##############################
                # Update running paths
                ##############################
                for idx in range(n_envs):
                    for t in range(len(seg_obs)):
                        if running_paths[idx] is None:
                            running_paths[idx] = dict(
                                observations=[],
                                actions=[],
                                rewards=[],
                                env_infos=[],
                                agent_infos=[],
                            )

                        running_paths[idx]["observations"].append(seg_obs[t][idx])
                        running_paths[idx]["actions"].append(seg_actions[t][idx])
                        running_paths[idx]["rewards"].append(seg_rewards[t][idx])
                        running_paths[idx]["env_infos"].append(
                            {k: v[idx] for k, v in seg_env_infos[t].items()}
                        )
                        running_paths[idx]["agent_infos"].append(
                            {k: v.data[idx] for k, v in seg_agent_infos[t].items()}
                        )
                        if seg_dones[t][idx]:
                            paths.append(dict(
                                actions=running_paths[idx]["actions"],
                                rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                                env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                                agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                            ))
                            running_paths[idx] = None

                seg_obs = []
                seg_actions = []
                seg_rewards = []
                seg_dones = []
                seg_env_infos = []
                seg_agent_infos = []

            seg_obs.append(obs)
            seg_actions.append(self.env.action_space.flatten_n(actions))
            seg_rewards.append(rewards)
            seg_dones.append(dones)
            seg_env_infos.append(env_infos)
            seg_agent_infos.append(agent_info_vars)

            self.policy.reset(dones)

            obs = next_obs

            if self.global_t.value - last_log_t >= self.epoch_length:
                if worker_id == 0:
                    # if len(paths) > 0:
                    self.log_diagnostics(
                        epoch=self.global_t.value // self.epoch_length,
                        global_t=self.global_t.value,
                        local_t=local_t,
                        paths=paths,
                    )
                    self.loggings = defaultdict(list)
                    paths = []
                    last_log_t = self.global_t.value
                    # sync local policy in case it spent too much time on logging
                    self.policy.set_param_values_from(self.shared_policy)

    def train(self):
        # launch workers
        seed = np.random.randint(np.iinfo(np.uint32).max)
        barrier = mp.Barrier(parties=max(1, self.n_parallel))
        if self.n_parallel <= 1:
            self.worker_train(worker_id=0, worker_seed=seed, barrier=barrier)
        else:
            processes = []
            for worker_id in range(self.n_parallel):
                process = mp.Process(
                    target=self.worker_train,
                    kwargs=dict(worker_id=worker_id, worker_seed=worker_id + seed, barrier=barrier)
                )
                processes.append(process)
            for p in processes:
                p.start()
            for p in processes:
                p.join()

    def log_diagnostics(self, epoch, global_t, local_t, paths):
        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('GlobalT', global_t)
        logger.record_tabular('LocalT', local_t)

        loggings = self.loggings

        logger.record_tabular('NumTrajs', len(paths))

        if len(paths) > 0:
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
            ent = self.policy.distribution.entropy(agent_infos)
        else:
            ent = [np.nan]
        logger.record_tabular('Entropy', np.mean(ent))
        logger.record_tabular('Perplexity', np.mean(np.exp(ent)))
        logger.record_tabular_misc_stat('TrajLen', [len(p["rewards"]) for p in paths], placement='front')

        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])
        undiscounted_returns = [np.sum(path["rewards"]) for path in paths]

        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')

        for key, vals in sorted(loggings.items()):
            logger.record_tabular(key, np.mean(vals))
        logger.record_tabular('PolicyParamNorm', np.linalg.norm(self.policy.get_param_values()))
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        logger.dump_tabular()
