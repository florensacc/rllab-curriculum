import pickle
from collections import defaultdict

import numpy as np
import chainer.functions as F
import chainer

from sandbox.rocky.chainer.misc import tensor_utils
from rllab.misc import logger
from rllab.misc import special
from rllab.misc import ext
from sandbox.rocky.chainer.algos_ref import copy_param
import multiprocessing as mp


class A3C(object):
    """
    Start working on an A3C implementation integrated with rllab's interface
    """

    def __init__(
            self,
            env,
            policy,
            optimizer,
            shared_params,
            shared_states,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1.,
            update_frequency=5,
            epoch_length=1000,
            n_epochs=1000,
            n_parallel=None,
            normalize_momentum=0.9,
            scale_reward=1.0,
            normalize_adv=True,
            normalize_vf=True,
            max_grad_norm=40.0,
            policy_loss_coeff=1.,
            vf_loss_coeff=0.5,
            entropy_bonus_coeff=1e-2,
    ):
        self.env = env
        self.policy = policy
        self.update_frequency = update_frequency
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.shared_params = shared_params
        self.shared_states = shared_states
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
            last_done,
    ):
        ##############################
        # Process data needed for optimization
        ##############################

        seg_rewards = np.asarray(seg_rewards) * self.scale_reward
        seg_dones = np.asarray(seg_dones)
        seg_vf_preds = np.asarray(
            [x['vf'].data.flatten() for x in seg_agent_infos + [next_agent_info]]
        ).flatten()

        seg_returns = np.empty_like(seg_rewards)
        for t in reversed(range(len(seg_obs))):
            if t == len(seg_obs) - 1:
                if last_done:
                    seg_returns[t] = 0
                else:
                    seg_returns[t] = seg_vf_preds[t + 1]
            else:
                seg_returns[t] = seg_rewards[t] + self.discount * self.gae_lambda * (1 - seg_dones[t]) * \
                                                  seg_returns[t + 1]

        seg_advs = seg_returns - seg_vf_preds[:-1]

        seg_actions = chainer.Variable(np.asarray(seg_actions))
        seg_dist_infos = dict()
        for k in self.policy.distribution.dist_info_keys:
            infos = [info[k] for info in seg_agent_infos]
            seg_dist_infos[k] = F.stack(infos)

        seg_vf_vars = F.flatten(F.stack([info['vf'] for info in seg_agent_infos]))

        ##############################
        # Compute objective and apply gradients
        ##############################

        N = seg_actions.shape[0]

        logli = self.policy.distribution.log_likelihood_sym(seg_actions, seg_dist_infos)

        policy_loss = -F.sum(logli * seg_advs) / N

        vf_loss = F.sum(F.square(seg_vf_vars - seg_returns)) / N

        entropy_loss = -F.sum(self.policy.distribution.entropy_sym(seg_dist_infos)) / N
        joint_loss = self.policy_loss_coeff * policy_loss + self.vf_loss_coeff * vf_loss + self.entropy_bonus_coeff * \
                                                                                           entropy_loss

        self.policy.zerograds()
        #
        # prev_grad = self.policy.get_grad_values()
        # (self.policy_loss_coeff * policy_loss).backward()
        # new_grad = self.policy.get_grad_values()
        # policy_grad_delta = np.linalg.norm(new_grad - prev_grad)
        # prev_grad = new_grad
        #
        # ((self.vf_loss_coeff) * vf_loss).backward()
        # new_grad = self.policy.get_grad_values()
        # vf_grad_delta = np.linalg.norm(new_grad - prev_grad)
        # prev_grad = new_grad
        #
        # (self.entropy_bonus_coeff * entropy_loss).backward()
        # new_grad = self.policy.get_grad_values()
        # ent_grad_delta = np.linalg.norm(new_grad - prev_grad)
        joint_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_policy.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_policy, source_link=self.policy)

        self.optimizer.update()

        copy_param.copy_param(target_link=self.policy,
                              source_link=self.shared_policy)

        self.loggings["PolicyLoss"].append(policy_loss.data)
        self.loggings["VfLoss"].append(vf_loss.data)
        self.loggings["EntropyLoss"].append(entropy_loss.data)
        self.loggings["JointLoss"].append(joint_loss.data)
        self.loggings["AbsVfPred"].append(np.mean(np.abs(seg_vf_vars.data)))
        self.loggings["AbsVfTarget"].append(np.mean(np.abs(seg_returns)))
        self.loggings["MaxVfTarget"].append(np.max(np.abs(seg_returns)))
        self.loggings["MinVfTarget"].append(np.min(np.abs(seg_returns)))
        # self.loggings["PolicyGradNorm"].append(policy_grad_delta)
        # self.loggings["VfGradNorm"].append(vf_grad_delta)
        # self.loggings["EntGradNorm"].append(ent_grad_delta)
        # self.loggings["RunningAdvMean"].append(self.running_adv_mean.value)
        # self.loggings["RunningAdvStd"].append(self.running_adv_std.value)
        # self.loggings["RunningVfMean"].append(self.running_vf_mean.value)
        # self.loggings["RunningVfStd"].append(self.running_vf_std.value)

    def worker_train(self, worker_id, worker_seed):
        ext.set_seed(worker_seed)

        self.shared_policy = self.policy
        self.policy = pickle.loads(pickle.dumps(self.shared_policy))
        from sandbox.rocky.chainer.algos_ref import async
        async.set_shared_params(self.shared_policy, self.shared_params)
        async.set_shared_states(self.optimizer, self.shared_states)
        copy_param.copy_param(target_link=self.policy,
                              source_link=self.shared_policy)

        obs = self.env.reset()
        self.policy.reset()

        action_var, agent_info_vars = self.policy.get_action_sym(obs)

        local_t = 0
        last_log_t = 0

        seg_obs = []
        seg_actions = []
        seg_rewards = []
        seg_dones = []
        seg_env_infos = []
        seg_agent_infos = []

        paths = []
        running_path = None

        while self.global_t.value < self.n_epochs * self.epoch_length:

            local_t += 1
            with self.global_t.get_lock():
                self.global_t.value += 1
            action = action_var.data

            next_obs, reward, done, env_info = self.env.step(action=action)
            next_action_var, next_agent_info_vars = self.policy.get_action_sym(next_obs)

            seg_obs.append(obs)
            seg_actions.append(self.env.action_space.flatten(action))
            seg_rewards.append(reward)
            seg_dones.append(done)
            seg_env_infos.append(env_info)
            seg_agent_infos.append(agent_info_vars)

            if len(seg_obs) >= self.update_frequency or done:

                self.update_once(
                    worker_id=worker_id,
                    seg_obs=seg_obs,
                    seg_actions=seg_actions,
                    seg_rewards=seg_rewards,
                    seg_dones=seg_dones,
                    seg_env_infos=seg_env_infos,
                    seg_agent_infos=seg_agent_infos,
                    next_agent_info=next_agent_info_vars,
                    last_done=done,
                )

                ##############################
                # Update running paths
                ##############################
                for t in range(len(seg_obs)):
                    if running_path is None:
                        running_path = dict(
                            actions=[],
                            rewards=[],
                            env_infos=[],
                            agent_infos=[],
                        )

                    # running_path["observations"].append(seg_obs[t])
                    running_path["actions"].append(seg_actions[t])
                    running_path["rewards"].append(seg_rewards[t])
                    running_path["env_infos"].append({k: v.data for k, v in seg_env_infos[t].items()})
                    running_path["agent_infos"].append({k: v.data for k, v in seg_agent_infos[t].items()})
                    if seg_dones[t]:
                        # import ipdb; ipdb.set_trace()
                        paths.append(dict(
                            actions=running_path["actions"],
                            rewards=tensor_utils.stack_tensor_list(running_path["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(running_path["env_infos"]),
                            agent_infos=tensor_utils.stack_tensor_dict_list(running_path["agent_infos"]),
                        ))
                        running_path = None

                seg_obs = []
                seg_actions = []
                seg_rewards = []
                seg_dones = []
                seg_env_infos = []
                seg_agent_infos = []

            if done:
                obs = self.env.reset()
                self.policy.reset()
                next_action_var, next_agent_info_vars = self.policy.get_action_sym(obs)
            else:
                obs = next_obs

            action_var = next_action_var
            agent_info_vars = next_agent_info_vars

            if self.global_t.value - last_log_t >= self.epoch_length:
                if worker_id == 0:
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
                    copy_param.copy_param(target_link=self.policy,
                                          source_link=self.shared_policy)

    def train(self):
        # launch workers
        seed = np.random.randint(np.iinfo(np.uint32).max)
        if self.n_parallel <= 1:
            self.worker_train(worker_id=0, worker_seed=seed)
        else:
            processes = []
            for worker_id in range(self.n_parallel):
                process = mp.Process(
                    target=self.worker_train,
                    kwargs=dict(worker_id=worker_id, worker_seed=worker_id + seed)
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
