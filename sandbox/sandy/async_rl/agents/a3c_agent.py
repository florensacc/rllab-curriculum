import copy
from logging import getLogger
import os, sys
import time
import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F

from sandbox.sandy.async_rl.agents.base import Agent
from sandbox.sandy.async_rl.utils import chainer_utils
from sandbox.sandy.async_rl.utils.nonbias_weight_decay import NonbiasWeightDecay
from sandbox.sandy.async_rl.networks import dqn_head, v_function
from sandbox.sandy.async_rl.policies import policy
from sandbox.sandy.async_rl.utils.init_like_torch import init_like_torch
from sandbox.sandy.async_rl.utils.rmsprop_async import RMSpropAsync
from sandbox.sandy.async_rl.shareable.base import Shareable
from sandbox.sandy.async_rl.utils.picklable import Picklable

logger = getLogger(__name__)
import rllab.misc.logger as mylogger


class A3CModel(chainer.Link):

    def pi_and_v(self, state, keep_same_state=False):
        """
        keep_same_state: maintain the hidden states of RNN, useful for just evaluating but not moving forward in time
        """
        raise NotImplementedError()

    def reset_state(self):
        pass

    def unchain_backward(self):
        pass

class A3CFF(chainer.ChainList, A3CModel):

    def __init__(self, n_actions, shared_weights=True, img_size=84):
        self.shared_weights = shared_weights
        if shared_weights:
            self.head = dqn_head.NIPSDQNHead(img_size=img_size) # observation -> feature
            self.pi = policy.FCSoftmaxPolicy(
                self.head.n_output_channels, n_actions)
            self.v = v_function.FCVFunction(self.head.n_output_channels)
            super().__init__(self.head, self.pi, self.v)
        else:
            self.pi_head = dqn_head.NIPSDQNHead(img_size=img_size)
            self.pi = policy.FCSoftmaxPolicy(
                self.pi_head.n_output_channels, n_actions)
            self.v_head = dqn_head.NIPSDQNHead(img_size=img_size)
            self.v = v_function.FCVFunction(self.v_head.n_output_channels)
            super().__init__(self.pi_head, self.pi, self.v_head, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        if self.shared_weights:
            out = self.head(state)
            return self.pi(out), self.v(out)
        else:
            return self.pi(self.pi_head(state)), self.v(self.v_head(state))


class A3CLSTM(chainer.ChainList, A3CModel):

    def __init__(self, n_actions, img_size=84):
        self.head = dqn_head.NIPSDQNHead(img_size=img_size)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super().__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()



class A3CAgent(Agent,Shareable,Picklable):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """

    def __init__(self,
                 n_actions,
                 model_type="ff",
                 optimizer_type="rmsprop_async",
                 optimizer_args=dict(lr=7e-4,eps=1e-1,alpha=0.99),
                 optimizer_hook_args=dict(
                    gradient_clipping=40,
                 ),
                 t_max=5, gamma=0.99, beta=1e-2,
                 process_id=0, clip_reward=True,
                 keep_loss_scale_same=False,
                 bonus_evaluator=None,
                 bonus_count_target="image",
                 phase="Train",
                 sync_t_gap_limit=np.inf,
                 shared_weights=True,
                 img_size=84
                 ):
        self.init_params = locals()
        self.init_params.pop('self')

        # Globally shared model
        if model_type == "ff":
            self.shared_model = A3CFF(n_actions,shared_weights,img_size=img_size)
        elif model_type == "lstm":
            self.shared_model == A3CLSTM(n_actions,img_size=img_size)
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

        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)

        self.t_max = t_max # maximum time steps before sending gradient update
        self.gamma = gamma # discount
        self.beta = beta # coeff for entropy bonus
        self.process_id = process_id
        self.clip_reward = clip_reward
        self.keep_loss_scale_same = keep_loss_scale_same

        self.bonus_evaluator = bonus_evaluator
        self.bonus_count_target = bonus_count_target

        self.phase = phase
        self.sync_t_gap_limit = sync_t_gap_limit

        self.t = 0
        self.t_start = 0
        self.last_sync_t = 0
        # they are dicts because the time index does not reset after finishing a traj
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_actions = {}
        self.past_rewards = {}
        self.past_values = {}
        self.past_extra_infos = {}
        self.epoch_entropy_list = []
        self.epoch_path_len_list = [0]
        self.epoch_effective_return_list = [0] # the return the agent truly sees
        self.epoch_adv_loss_list = []
        self.epoch_entropy_loss_list = []
        self.epoch_v_loss_list = []
        self.epoch_sync_t_gap_list = []
        self.cur_path_len = 0
        self.cur_path_effective_return = 0
        self.unpicklable_list = ["shared_params","shared_model"]

    def prepare_sharing(self):
        self.shared_params = dict(
            model_params=chainer_utils.extract_link_params(self.shared_model),
        )
        if self.bonus_evaluator is not None:
            self.bonus_evaluator.prepare_sharing()

    def process_copy(self):
        new_agent = A3CAgent(**self.init_params)
        chainer_utils.set_link_params(
            new_agent.shared_model,
            self.shared_params["model_params"],
        )
        new_agent.sync_parameters()
        if self.bonus_evaluator is not None:
            new_agent.bonus_evaluator = self.bonus_evaluator.process_copy()
        new_agent.shared_params = self.shared_params

        return new_agent

    def sync_parameters(self):
        chainer_utils.copy_link_param(
            target_link=self.model,
            source_link=self.shared_model,
        )

    def preprocess(self,state):
        #assert state[0].dtype == np.uint8
        #processed_state = (np.asarray(state, dtype=np.float32) / 255.0 * 2.0) - 1.0
        processed_state = np.asarray(state, dtype=np.float32)
        return processed_state

    def act(self, state, reward, is_state_terminal, extra_infos=dict(),global_vars=dict(),training_args=dict()):
        # reward shaping
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)
        self.past_rewards[self.t - 1] = reward

        if not is_state_terminal:
            statevar = chainer.Variable(np.expand_dims(self.preprocess(state), 0))

        # record the time elapsed since last model synchroization
        # if the time is too long, we may discard the current update and synchronize instead
        if self.phase == "Train":
            sync_t_gap = global_vars["global_t"].value - self.last_sync_t
            not_delayed = sync_t_gap < self.sync_t_gap_limit

        ready_to_commit = self.phase == "Train" and (
            (is_state_terminal and self.t_start < self.t) \
                or self.t - self.t_start == self.t_max)
        # start computing gradient and synchronize model params
        # avoid updating model params during testing
        if ready_to_commit:
            assert self.t_start < self.t

            # assign bonus rewards
            if self.bonus_evaluator is not None:
                if self.bonus_count_target == "image":
                    count_targets = np.asarray([
                        self.past_states[i].data[0]
                        for i in range(self.t_start, self.t)
                    ])
                elif self.bonus_count_target == "ram":
                    count_targets = np.asarray([
                        self.past_extra_infos[i]["ram_state"]
                        for i in range(self.t_start, self.t)
                    ])
                else:
                    raise NotImplementedError
                bonus_rewards = self.bonus_evaluator.update_and_evaluate(count_targets)
                for i in range(self.t_start, self.t):
                    self.past_rewards[i] += bonus_rewards[i - self.t_start]
            self.cur_path_effective_return += np.sum([
                self.past_rewards[i] for i in range(self.t_start, self.t)
            ])


            # bootstrap total rewards for a final non-terminal state
            if is_state_terminal:
                R = 0
            else:
                _, vout = self.model.pi_and_v(statevar, keep_same_state=True)
                R = float(vout.data)

            adv_loss = 0
            entropy_loss = 0
            v_loss = 0
            # WARNING: the losses are accumulated instead of averaged over time steps
            for i in reversed(range(self.t_start, self.t)):
                R *= self.gamma
                R += self.past_rewards[i]
                v = self.past_values[i]
                if self.process_id == 0:
                    logger.debug('s:%s v:%s R:%s',
                                 self.past_states[i].data.sum(), v.data, R)
                advantage = R - v
                # Accumulate gradients of policy
                log_prob = self.past_action_log_prob[i]
                entropy = self.past_action_entropy[i]

                # Log probability is increased proportionally to advantage
                adv_loss -= log_prob * float(advantage.data)
                # Entropy is maximized
                entropy_loss -= self.beta * entropy
                # Accumulate gradients of value function
                v_loss += (v - R) ** 2 / 2

            # Normalize the loss of sequences truncated by terminal states
            if self.keep_loss_scale_same and \
                    self.t - self.t_start < self.t_max:
                factor = self.t_max / (self.t - self.t_start)
                adv_loss *= factor
                entropy_loss *= factor
                v_loss *= factor
            pi_loss = adv_loss + entropy_loss

            if self.process_id == 0:
                logger.debug('adv_loss:%s, entropy_loss:%s, pi_loss:%s, v_loss:%s', adv_loss.data, entropy_loss.data, pi_loss.data, v_loss.data)

            # note that policy and value share the same lower layers
            total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

            # Update the globally shared model
            if not_delayed:
                # Compute gradients using thread-specific model
                self.model.zerograds()
                total_loss.backward()
                # Copy the gradients to the globally shared model
                self.shared_model.zerograds()
                chainer_utils.copy_link_grad(
                    target_link=self.shared_model,
                    source_link=self.model
                )
                self.optimizer.update()
            else:
                mylogger.log("Process %d banned from commiting gradient update from %d time steps ago."%(self.process_id,sync_t_gap))

            # log the losses
            self.epoch_adv_loss_list.append(adv_loss.data)
            self.epoch_entropy_loss_list.append(entropy_loss.data)
            self.epoch_v_loss_list.append(v_loss.data)

            self.sync_parameters()
            self.epoch_sync_t_gap_list.append(sync_t_gap)
            self.last_sync_t = global_vars["global_t"].value
            self.model.unchain_backward()

            # initialize stats for a new traj
            self.past_action_log_prob = {}
            self.past_action_entropy = {}
            self.past_states = {}
            self.past_actions = {}
            self.past_rewards = {}
            self.past_values = {}
            self.past_extra_infos = {}

            self.t_start = self.t

        # store traj info and return action
        if not is_state_terminal:
            pout, vout = self.model.pi_and_v(statevar)
            action = pout.action_indices[0]
            if self.phase == "Train":
                self.past_states[self.t] = statevar
                self.past_actions[self.t] = action
                self.past_action_log_prob[self.t] = pout.sampled_actions_log_probs
                self.past_action_entropy[self.t] = pout.entropy
                self.past_values[self.t] = vout
                self.past_extra_infos[self.t] = extra_infos
                self.t += 1
                if self.process_id == 0:
                    logger.debug('t:%s entropy:%s, probs:%s',
                                 self.t, pout.entropy.data, pout.probs.data)
                self.epoch_entropy_list.append(pout.entropy.data)
                self.cur_path_len += 1
            else:
                self.model.unchain_backward()
            return action
        else:
            self.epoch_path_len_list.append(self.cur_path_len)
            self.cur_path_len = 0
            self.epoch_effective_return_list.append(self.cur_path_effective_return)
            self.cur_path_effective_return = 0
            self.model.reset_state()
            return None


    def finish_epoch(self,epoch,log):
        if self.bonus_evaluator is not None:
            self.bonus_evaluator.finish_epoch(epoch=epoch,log=log)
        if log:
            mylogger.record_tabular("_ProcessID", self.process_id)
            mylogger.record_tabular("_LearningRate", self.optimizer.lr)
            entropy = np.average(self.epoch_entropy_list)
            mylogger.record_tabular("_Entropy",entropy)
            mylogger.record_tabular("_Perplexity",np.exp(entropy))
            mylogger.record_tabular_misc_stat("_PathLen",self.epoch_path_len_list)
            mylogger.record_tabular_misc_stat("_EffectiveReturn",self.epoch_effective_return_list)
            mylogger.record_tabular_misc_stat(
                "_AdvLoss",
                self.epoch_adv_loss_list,
            )
            mylogger.record_tabular_misc_stat(
                "_EntropyLoss",
                self.epoch_entropy_loss_list,
            )
            mylogger.record_tabular_misc_stat(
                "_ValueLoss",
                self.epoch_v_loss_list,
            )
            mylogger.record_tabular_misc_stat(
                "_SyncTimeGap",
                self.epoch_sync_t_gap_list,
            )
        self.epoch_entropy_list = []
        self.epoch_effective_return_list = [0]
        self.epoch_path_len_list = [0]
        self.epoch_adv_loss_list = []
        self.epoch_entropy_loss_list = []
        self.epoch_v_loss_list = []
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



    def update_params(self, global_vars, training_args):
        if self.phase == "Train":
            global_t = global_vars["global_t"].value
            total_steps = training_args["total_steps"]
            self.optimizer.lr = self.init_lr * (total_steps - global_t) / total_steps
