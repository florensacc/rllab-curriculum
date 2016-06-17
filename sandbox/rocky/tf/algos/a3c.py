from __future__ import print_function
from __future__ import absolute_import
from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.core.serializable import Serializable
import tensorflow as tf
import threading
import cPickle as pickle
import numpy as np


def run_algo(algo, worker_id):
    algo.worker_train(worker_id)


class A3C(RLAlgorithm, Serializable):
    def __init__(
            self,
            env,
            policy,
            max_path_length=100,
            batch_size=32,
            scale_reward=1.,
            optimizer=None,
            optimizer_args=None):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.scale_reward = scale_reward
        self.worker_envs = None
        self.T = 0

    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - tf.reduce_sum(logli * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)
        else:
            surr_obj = - tf.reduce_mean(logli * advantage_var)
            mean_kl = tf.reduce_mean(kl)
            max_kl = tf.reduce_max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(surr_obj, target=self.policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    def worker_train(self, worker_id):
        # create a worker-specific copy
        env = self.worker_envs[worker_id]

        terminal = False
        obs = self.env.reset()
        t = 0

        observations = []
        next_observations = []
        actions = []
        rewards = []
        terminals = []


        while True:
            if terminal:
                obs = self.env.reset()
                t = 0
                # es.reset()
            # increment global counter
            self.T += 1

            # get action
            action, action_info = self.policy.get_action(obs)
            next_obs, reward, terminal, env_info = self.env.step(action)

            if not terminal and t >= self.max_path_length:
                terminal = True
            else:
                observations.append(self.env.observation_space.flatten(obs))
                next_observations.append(self.env.observation_space.flatten(next_obs))
                actions.append(self.env.action_space.flatten(action))
                rewards.append(reward * self.scale_reward)
                terminals.append(terminal)
            obs = next_obs
            if len(observations) >= self.batch_size:
                observations = np.array(observations)
                next_observations = np.array(next_observations)
                actions = np.array(actions)
                rewards = np.array(rewards)
                terminals = np.array(terminals)

                observations = []
                next_observations = []
                actions = []
                rewards = []
                terminals = []

            pass
        print("lala")

    def train(self):
        self.init_opt()
        n_workers = 4
        self.worker_envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_workers)]

        workers = []

        for idx in range(n_workers):
            # worker_env =
            t = threading.Thread(target=run_algo, args=(self, idx))
            t.start()
            workers.append(t)
        for worker in workers:
            worker.join()
