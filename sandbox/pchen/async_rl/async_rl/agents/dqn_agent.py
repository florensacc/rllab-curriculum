import copy
from enum import Enum
from logging import getLogger
import os
import time

import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F
import chainer.links as L

from sandbox.pchen.async_rl.async_rl.utils import chainer_utils
from sandbox.pchen.async_rl.async_rl.utils.init_like_torch import init_like_torch
from sandbox.pchen.async_rl.async_rl.agents.base import Agent
from sandbox.pchen.async_rl.async_rl.networks.dqn_head import NIPSDQNHead, NatureDQNHead
from sandbox.pchen.async_rl.async_rl.shareable.base import Shareable
from sandbox.pchen.async_rl.async_rl.utils.picklable import Picklable
from sandbox.pchen.async_rl.async_rl.utils.rmsprop_async import RMSpropAsync

from rllab.misc import logger as mylogger

logger = getLogger(__name__)


class DQNModel(chainer.Link):
    def compute_qs(self,state):
        raise NotImplementedError

    def reset_state(self):
        pass


class DQNNIPSModel(chainer.ChainList, DQNModel):
    def __init__(self, n_actions):
        self.head = NIPSDQNHead()
        self.head_to_q = L.Linear(self.head.n_output_channels, n_actions)
        super().__init__(self.head, self.head_to_q)
        init_like_torch(self)

    def compute_qs(self, state):
        return self.head_to_q(self.head(state))

class DQNNatureModel(chainer.ChainList, DQNModel):
    def __init__(self, n_actions):
        self.head = NatureDQNHead()
        self.head_to_q = L.Linear(self.head.n_output_channels, n_actions)
        super().__init__(self.head, self.head_to_q)
        init_like_torch(self)

    def compute_qs(self, state):
        return self.head_to_q(self.head(state))

class Bellman(Enum):
    q = 1
    sarsa = 2


class DQNAgent(Agent,Shareable,Picklable):
    """
    Asynchronous n-step Q-learning (n-Step Q-learning)
    See http://arxiv.org/abs/1602.01783
    Notice that the target network is globally shared. To reduce computation time we may try keeping local copies instead.
    """

    def __init__(
            self,
            env,
            bellman=Bellman.q,
            model_type="nips",
            optimizer_type="rmsprop_async",
            optimizer_args=None,
            optimizer_hook_args=None,
            t_max=5,
            gamma=0.99,
            beta=1e-2,
            eps_start=1.0,
            eps_end=0.1,
            eps_anneal_time=4 * 10 ** 6,
            eps_test=None, # eps used for testing regardless of training eps
            target_update_frequency=40000,
            process_id=0, clip_reward=True,
            keep_loss_scale_same=False,
            bonus_evaluator=None,
            bonus_count_target="image",
            phase="Train",
            sync_t_gap_limit=np.inf,
            share_model=False,
            sample_eps=False,
    ):
        if optimizer_args is None:
            optimizer_args = dict(lr=7e-4, eps=1e-1, alpha=0.99)
        if optimizer_hook_args is None:
            optimizer_hook_args = dict(
                gradient_clipping=40,
            )
        self.init_params = locals()
        self.init_params.pop('self')

        self.env = env
        self.bellman = bellman
        self.share_model = share_model

        action_space = env.action_space

        # Globally shared model
        if model_type == "nips":
            self.shared_model = DQNNIPSModel(action_space.n)
        elif model_type == "nature":
            raise Exception("not implement")
            self.shared_model = DQNNIPSModel(action_space.n)
        else:
            raise NotImplementedError

        # Optimizer
        if optimizer_type == "rmsprop_async":
            self.optimizer = RMSpropAsync(**optimizer_args)
        else:
            raise NotImplementedError
        self.optimizer.setup(self.shared_model)
        if "gradient_clipping" in optimizer_hook_args:
            self.optimizer.add_hook(chainer.optimizer.GradientClipping(
                optimizer_hook_args["gradient_clipping"]
            ))
        if "weight_decay" in optimizer_hook_args:
            self.optimizer.add_hook(NonbiasWeightDecay(
                optimizer_hook_args["weight_decay"]
            ))
        self.init_lr = self.optimizer.lr

        self.shared_target_model = copy.deepcopy(self.shared_model)
        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.t_max = t_max # maximum time steps before sending gradient update
        self.gamma = gamma # discount
        self.process_id = process_id
        self.clip_reward = clip_reward
        self.keep_loss_scale_same = keep_loss_scale_same
        self.eps_start = eps_start
        if not sample_eps:
            self.eps_end = eps_end
        else:
            self.eps_end = np.random.choice(
                a=[0.1, 0.01, 0.5],
                p=[0.4, 0.3, 0.3]
            )
        self.eps_test = eps_test
        self.target_update_frequency = target_update_frequency
        self.eps_anneal_time = eps_anneal_time
        self.bonus_evaluator = bonus_evaluator
        self.bonus_count_target = bonus_count_target
        self.phase = phase
        self.sync_t_gap_limit = sync_t_gap_limit

        self.eps = self.eps_start
        self.t = 0
        self.t_start = 0
        self.last_sync_t = 0
        # they are dicts because the time index does not reset after finishing a traj
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_actions = {}
        self.past_rewards = {}
        self.past_qvalues = {}
        self.past_extra_infos = {}
        self.epoch_td_loss_list = []
        self.epoch_q_list = []
        self.epoch_path_len_list = [0]
        self.epoch_sync_t_gap_list = []
        self.cur_path_len = 0
        self.epoch_effective_return_list = [0] # the return the agent truly sees
        self.cur_path_effective_return = 0
        self.unpicklable_list = ["shared_params","shared_model","shared_target_model"]

    def prepare_sharing(self):
        self.shared_params = dict(
            model_params=chainer_utils.extract_link_params(self.shared_model),
            target_model_params=chainer_utils.extract_link_params(self.shared_target_model),
        )

    def process_copy(self):
        new_agent = DQNAgent(**self.init_params)
        chainer_utils.set_link_params(
            new_agent.shared_model,
            self.shared_params["model_params"],
        )
        chainer_utils.set_link_params(
            new_agent.shared_target_model,
            self.shared_params["target_model_params"],
        )
        new_agent.sync_parameters(init=True)
        new_agent.shared_params = self.shared_params
        new_agent.eps = self.eps # important for testing!

        return new_agent

    def sync_parameters(self, init=False):
        if (init) or (not self.share_model):
            chainer_utils.copy_link_param(
                source_link=self.shared_model,
                target_link=self.model,
                deep=not self.share_model,
            )

    def preprocess(self,state):
        # delegate this to env wrapper
        return state

    def update_params(self, global_vars, training_args):
        if self.phase == "Train":
            # change learning rate
            global_t = global_vars["global_t"].value
            total_steps = training_args["total_steps"]
            self.optimizer.lr = self.init_lr * (total_steps - global_t) / total_steps

            # change exploration eps
            if global_t < self.eps_anneal_time:
                a = (self.eps_end - self.eps_start) / self.eps_anneal_time
                b = self.eps_start
                self.eps = a * global_t + b
            else:
                self.eps = self.eps_end

            # update target network
            if global_t % self.target_update_frequency == 0:
                mylogger.log("Updating target network",color="yellow")
                chainer_utils.copy_link_param(
                    target_link=self.shared_target_model,
                    source_link=self.shared_model,
                )


    def act(
            self, state, reward, is_state_terminal,
            extra_infos=None, global_vars=None,
    ):
        if extra_infos is None:
            extra_infos = dict()
        if global_vars is None:
            global_vars = dict()
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)
        self.past_rewards[self.t - 1] = reward

        if not is_state_terminal:
            statevar = chainer.Variable(np.expand_dims(self.preprocess(state), 0))

        ready_to_commit = self.phase == "Train" and (
            (is_state_terminal and self.t_start < self.t) or
            (self.t - self.t_start == self.t_max)
        )

        # start computing gradient and synchronize model params
        # avoid updating model params during testing
        if ready_to_commit:
            assert self.t_start < self.t

            # assign bonus rewards
            if is_state_terminal:
                R = 0
            else:
                # bootstrap from target network
                qs = self.shared_target_model.compute_qs(statevar)[0] # fixed [0]
                if self.bellman == Bellman.q:
                    R = float(np.amax(qs.data))
                elif self.bellman == Bellman.sarsa:
                    R = float(qs.data[self.past_actions[self.t - 1]])
                else:
                    raise NotImplementedError

            loss = 0
            for i in reversed(range(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                q = self.past_qvalues[i]
                cur_loss = 0.5 * (q - R) ** 2
                loss += cur_loss
                self.epoch_td_loss_list.append(cur_loss.data)

            # Normalize the loss of sequences truncated by terminal states
            if self.keep_loss_scale_same and \
                    self.t - self.t_start < self.t_max:
                factor = self.t_max / (self.t - self.t_start)
                loss *= factor

            # record the time elapsed since last model synchroization
            # if the time is too long, we may discard the current update and synchronize instead
            sync_t_gap = global_vars["global_t"].value - self.last_sync_t
            not_delayed = sync_t_gap < self.sync_t_gap_limit
            if not_delayed:
                # Compute gradients using thread-specific model
                self.model.zerograds()
                loss.backward()
                if not self.share_model:
                    # Copy the gradients to the globally shared model
                    self.shared_model.zerograds()
                    chainer_utils.copy_link_grad(
                        target_link=self.shared_model,
                        source_link=self.model
                    )
                self.optimizer.update()
            else:
                mylogger.log("Process %d banned from commiting gradient update from %d time steps ago."%(self.process_id,sync_t_gap))

            self.sync_parameters()
            self.epoch_sync_t_gap_list.append(sync_t_gap)
            self.last_sync_t = global_vars["global_t"].value

            # initialize stats for a new traj
            self.past_states = {}
            self.past_actions = {}
            self.past_rewards = {}
            self.past_qvalues = {}
            self.past_extra_infos = {}

            self.t_start = self.t

        # store traj info and return action
        if not is_state_terminal:
            # choose an action
            qs = self.model.compute_qs(statevar)[0]
            # WARN: even when testing, do not just use argmax; it may get stuck at the beginning of Breakout (doing no_op)
            if self.phase == "Test" and self.eps_test is not None:
                eps = self.eps_test
            else:
                eps = self.eps
            if np.random.uniform() < eps:
                a = np.random.randint(low=0, high=len(qs.data))
            else:
                a = np.argmax(qs.data)

            # update info for training; doing this in testing will lead to insufficient memory
            if self.phase == "Train":
                # record the state to allow bonus computation
                self.past_states[self.t] = statevar
                self.past_actions[self.t] = a
                self.past_qvalues[self.t] = qs[a] # beware to record the variable (not just its data) to allow gradient computation
                self.past_extra_infos[self.t] = extra_infos
                self.epoch_q_list.append(float(qs[a].data))
                self.cur_path_len += 1
                self.t += 1
            return a
        else:
            self.epoch_path_len_list.append(self.cur_path_len)
            self.cur_path_len = 0
            self.epoch_effective_return_list.append(self.cur_path_effective_return)
            self.cur_path_effective_return = 0
            self.model.reset_state()
            return None


    def finish_epoch(self, epoch, log):
        if self.bonus_evaluator is not None:
            self.bonus_evaluator.finish_epoch(epoch=epoch,log=log)
        if log:
            mylogger.record_tabular("_ProcessID",self.process_id)
            mylogger.record_tabular("_LearningRate", self.optimizer.lr)
            mean_td_loss = np.average(self.epoch_td_loss_list)
            mylogger.record_tabular("_TDLossAverage",mean_td_loss)

            max_q = np.amax(self.epoch_q_list)
            mylogger.record_tabular("_MaxQ",max_q)

            mylogger.record_tabular("_Epsilon",self.eps)

            mylogger.record_tabular_misc_stat("_PathLen",self.epoch_path_len_list)

            mylogger.record_tabular_misc_stat("_EffectiveReturn",self.epoch_effective_return_list)
            mylogger.record_tabular_misc_stat(
                "_SyncTimeGap",
                self.epoch_sync_t_gap_list,
            )
        self.epoch_td_loss_list = []
        self.epoch_q_list = []
        self.epoch_path_len_list = [0]
        self.epoch_effective_return_list = [0]
        self.epoch_sync_t_gap_list = []

        if log:
            mylogger.log(
                "Process %d finishes epoch %d with logging."%(self.process_id,epoch),
                color="green"
            )
        else:
            mylogger.log(
                "Process %d finishes epoch %d without logging."%(self.process_id,epoch),
                color="green"
            )
