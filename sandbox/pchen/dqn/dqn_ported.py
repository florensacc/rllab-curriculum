import scipy
import numpy as np
import functools
from collections import defaultdict
import collections
import numpy as np, numpy.random as nr
import numpy as np
import time, sys
import pdb
import itertools, random
import math
import pickle as pickle
import scipy.optimize
import scipy.misc
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import time
import numpy as np
import itertools
import operator
import scipy
import scipy.optimize
import sys
import lasagne
import lasagne.layers as L
from IPython.display import SVG
from collections import defaultdict, OrderedDict
import argparse
import os
import json
import rllab.misc.logger as logger

from sandbox.pchen.dqn.utils.common import rgb2y, wrapped_conv, init_from_args, gen_updates, experiment_iter, \
    copy_from_params, set_to_params, ind_from_vecs


def preprocess(rgb, input_size):
    # note here we follow the code that's released with Nature article instead of the one described in Minh 2013
    # to y
    y = rgb2y(rgb)
    # downsample (bilinear)
    # note imresize rependes on PIL https://github.com/python-pillow/Pillow
    sampled = scipy.misc.imresize(y, input_size, interp='bilinear')
    normalized = (sampled - np.mean(sampled)) / 128
    return normalized

def update_from_memory_batch(
        q_func,
        Q_params,
        updater,
        memory,
        reward_mask,
        terminal_mask,
        action_mask,
        batch_inds,
        gamma,
        prev_param_vals,
        prepare_inputX
):
    bs = len(batch_inds)
    bnX = prepare_inputX(memory, batch_inds + 1)
    staged = copy_from_params(Q_params)
    set_to_params(prev_param_vals, Q_params)
    q_max_idx = np.argmax(q_func(bnX), axis=1)
    ys = memory[batch_inds, reward_mask].reshape(-1) + \
         (gamma * (q_func(bnX))[np.arange(bs), q_max_idx]) \
         * (1 - memory[batch_inds, terminal_mask]).reshape(-1)
    # reuse to save meaningless memory
    bnX = prepare_inputX(memory, batch_inds)
    set_to_params(staged, Q_params)
    return updater(bnX, (memory[batch_inds, action_mask]).astype('int32').ravel(), ys)

class DQNP(object):
    def __init__(
            self,
            env,
            gamma=0.99,
            optimizer="adam",
            learning_rate=1e-5,
            max_iter=1000000,# type=int,
            replay_min=50000,# type=int,
            network_update_freq=10000,# type=int,
            evaluation_freq=50000,# type=int,
            evaluation_len=10000,# type=int,
            exp_name="dqn",# type=str,
            out_dir="./",# type=str,
            checkpoint_freq=100000,# type=int,
            batch_size=32,# type=int,
            network=None,# type=str,
            network_args=None,# type=str,
            min_eps=0.01,
            algo=None,# type=str,
            terminate_per_life=False,# type=str2bool,
            temporal_frames=3,# type=int,
            load_params=None,# type=str,
            load_params_mask=None,# type=json.loads,
            train_params_mask=None,# type=json.loads,
            lbfgs_iters=10,# type=int,
            penalty_type="action",# type=str,
            test_batch_size=50,# type=int,
            dup_factor=1,# type=int,
            validation_batch_size=1000,# type=int,
            shuffle_update_order=False,# type=str2bool,
            penalty_rate=1.,
            eval_runs=5,# type=int,
            train_runs=1,# type=int,
            train_switch_freq=0,# type=int,
            n_processes=2,# type=int,
            red_factor=15,# type=int,
            update_slave=True,# type=str2bool,
            dropout_max=False,# type=str2bool,
            start_from=1,# type=int,
            memory_size=190000,# type=int,
            no_replay=False,# type=str2bool,
            dropout_rollout=False,# type=str2bool,
            load_expert_params=None,# type=str,
            regress_q_pi=False,# type=str2bool,
            select_max_by_current_q=False,# type=str2bool,
    ):
        go = locals()
        del go["self"]
        self.__dict__.update(go)

    def train(self):
        env = self.env
        action_space = env.action_space
        action_size = action_space.n
        reward_set = [0, 1]

        network_input_img_size = (84, 84)
        network_input_dim = (self.temporal_frames,) + network_input_img_size
        network_output_dim = action_size
        reward_dim = len(reward_set)

        # Q_X = T.tensor4("X")
        dqn_in = L.InputLayer(shape=(None,) + network_input_dim, )# input_var=Q_X)
        Q_X = dqn_in.input_var
        if self.network_args == "nature":
            dqn_conv1 = L.Conv2DLayer(dqn_in, 32, 8, stride=4, )
            dqn_conv2 = L.Conv2DLayer(dqn_conv1, 64, 4, stride=2, )
            dqn_conv3 = L.Conv2DLayer(dqn_conv2, 64, 3, stride=1, )
            dqn_fc1 = L.DenseLayer(dqn_conv3, 512, nonlinearity=lasagne.nonlinearities.rectify)
            dqn_out = L.DenseLayer(dqn_fc1, network_output_dim, nonlinearity=lasagne.nonlinearities.identity)
        else:
            dqn_conv1 = L.Conv2DLayer(dqn_in, 16, 8, stride=4, )
            dqn_conv2 = L.Conv2DLayer(dqn_conv1, 32, 4, stride=2, )
            dqn_fc1 = L.DenseLayer(dqn_conv2, 256, nonlinearity=lasagne.nonlinearities.rectify)
            dqn_out = L.DenseLayer(dqn_fc1, network_output_dim, nonlinearity=lasagne.nonlinearities.identity)

        Q_vals = L.get_output(dqn_out)
        Q_args = [Q_X]
        Q_vals_fn = theano.function(Q_args, Q_vals)

        a_inds = T.ivector("A_inds")
        Q_tgt_vals = T.vector("Q_tgt_vals")
        Q_selected_vals = Q_vals[T.arange(Q_X.shape[0]), a_inds]

        Q_loss_args = Q_args + [a_inds, Q_tgt_vals]
        Q_loss = T.sum(T.square(Q_selected_vals - Q_tgt_vals)) / T.cast(Q_X.shape[0], 'float32')
        Q_loss_fn = theano.function(Q_loss_args, Q_loss)

        Q_params = L.get_all_params(dqn_out)

        Q_updates = gen_updates(Q_loss, Q_params, self)

        Q_train_function = theano.function(Q_loss_args, Q_loss, updates=Q_updates)
        print("Compilation done")

        # memory :: Matrix StepId (Observation, Action, Terminal, Reward)
        memory_size = 190000
        observation_mask = slice(0, network_input_img_size[0] * network_input_img_size[1])
        action_mask = slice(observation_mask.stop, observation_mask.stop + 1)
        terminal_mask = slice(action_mask.stop, action_mask.stop + 1)
        reward_mask = slice(terminal_mask.stop, terminal_mask.stop + 1)

        memory = np.zeros((memory_size, reward_mask.stop), dtype='float16')

        def prepare_inputX(memory, idx):
            bs = len(idx)
            bnX = np.zeros((bs,) + network_input_dim, dtype='float32')
            for i, ind in zip(range(bs), idx):
                bnX[i] = memory[(ind - self.temporal_frames + 1):(ind + 1), observation_mask].reshape(network_input_dim)
            return bnX


        memory_i = 0
        t = time.time()
        period_i = 0
        period_plays = 0
        period_rewards = 0
        to_train_inds = []

        ob = env.reset()

        for step_i in range(self.max_iter):
            memory[memory_i, observation_mask] = preprocess(ob, network_input_img_size).ravel()

            eps = max(self.min_eps, 1 - 0.9 / 1e6 * step_i)

            if np.random.rand() < eps or memory_i < (self.temporal_frames-1):
                action = action_space.sample()
            else:
                bX = memory[(memory_i - self.temporal_frames + 1):(memory_i + 1), observation_mask].reshape(
                    (1,) + network_input_dim)
                action = np.argmax(Q_vals_fn(bX))

            ob, reward, done, _ = env.step(action)

            memory[memory_i, action_mask] = action
            memory[memory_i, terminal_mask] = int(done)
            memory[memory_i, reward_mask] = 0 if reward == 0 else (1 if reward > 0 else -1)
            memory_i = (memory_i + 1) % memory_size
            if done:
                period_plays += 1
                ob = env.reset()
            period_rewards += reward

            if step_i >= self.replay_min:
                # fixing behabior policy but approximate gradient steps
                for _ in range(self.dup_factor):
                    to_train_inds.append(
                        np.random.randint(
                            self.temporal_frames - 1,
                            min(step_i, memory_size - 1) - 1,
                            self.batch_size)
                    )

                if (step_i % self.network_update_freq) == 0:
                    this_stat = {}
                    this_stat['sampling_time'] = time.time() - t
                    t = time.time()
                    prev_param_vals = copy_from_params(Q_params)
                    errors = []
                    for train_i, sample_inds in enumerate(to_train_inds):
                        cost = update_from_memory_batch(
                            Q_vals_fn, Q_params, Q_train_function, memory,
                            reward_mask, terminal_mask, action_mask,
                            sample_inds, self.gamma, prev_param_vals, prepare_inputX)
                        errors.append(cost)
                    this_stat['q_err_mean'] = np.mean(errors)
                    this_stat['q_err_std'] = np.std(errors)
                    this_stat['optimization_time'] = time.time() - t
                    f_safe_period_plays = float(max(period_plays, 1))
                    this_stat['train_reward_avg'] = period_rewards / f_safe_period_plays
                    period_i += 1
                    period_plays *= 0
                    period_rewards *= 0
                    to_train_inds[:] = []
                    t *= 0.
                    t += time.time()

                    for k,v in list(this_stat.items()):
                        logger.record_tabular(k, v)
                logger.dump_tabular(with_prefix=False)

