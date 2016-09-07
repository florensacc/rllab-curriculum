

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import threading
import queue
import pickle as pickle
import numpy as np
import time
import multiprocessing
from rllab.misc import special
from rllab.misc import logger
from tensorflow.python.framework import ops


def run_algo(algo, worker_id, sess, run_event):
    try:
        with ops.default_session(sess):
            algo.worker_train(worker_id, run_event)
    except Exception as e:
        print(e)
        raise


def chain(f, g):
    def fun(x):
        return f(g(x))
    return fun


aggregate_perplexity = chain(np.exp, np.mean)


class A3C(RLAlgorithm, Serializable):
    def __init__(
            self,
            env,
            policy,
            critic_network,
            n_workers=None,
            max_epochs=1000,
            epoch_length=10000,
            max_path_length=100,
            batch_size=32,
            scale_reward=1.,
            discount=0.99,
            policy_optimizer=None,
            critic_optimizer=None,
    ):
        Serializable.quick_init(self, locals())

        if policy_optimizer is None:
            policy_optimizer = tf.train.AdamOptimizer()
        if critic_optimizer is None:
            critic_optimizer = tf.train.AdamOptimizer()
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        self.n_workers = n_workers
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.opt_info = None
        self.env = env
        self.policy = policy
        self.critic_network = critic_network
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.scale_reward = scale_reward
        self.worker_envs = None
        self.worker_stats_queue = queue.Queue()
        self.max_epochs = max_epochs
        self.epoch_length = epoch_length
        self.max_T = max_epochs * epoch_length
        self.T = 0
        self.discount = discount
        self.init_opt()

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
        returns_var = tensor_utils.new_tensor(
            name='returns',
            ndim=1 + is_recurrent,
            dtype=tf.float32
        )

        advantage_var = returns_var - tf.reshape(self.critic_network.predict_sym(obs_var), (-1,))

        dist = self.policy.distribution

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

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - tf.reduce_sum(logli * tf.stop_gradient(advantage_var) * valid_var) / tf.reduce_sum(valid_var)
        else:
            surr_obj = - tf.reduce_mean(logli * tf.stop_gradient(advantage_var))

        input_list = [obs_var, action_var, returns_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        critic_loss = tf.reduce_mean(tf.square(advantage_var))

        policy_train_op = self.policy_optimizer.minimize(surr_obj, var_list=self.policy.get_params(trainable=True))
        critic_train_op = self.critic_optimizer.minimize(critic_loss,
                                                         var_list=self.critic_network.get_params(trainable=True))

        f_train_policy = tensor_utils.compile_function(
            inputs=input_list,
            outputs=policy_train_op,
        )

        f_train_critic = tensor_utils.compile_function(
            inputs=input_list,
            outputs=critic_train_op,
        )

        self.opt_info = dict(
            f_train_policy=f_train_policy,
            f_train_critic=f_train_critic
        )

    def worker_train(self, worker_id, run_event):
        # create a worker-specific copy
        worker_env = self.worker_envs[worker_id]

        t = 0
        last_obs = None
        path_samples_data = []

        while run_event.is_set():

            last_obs, t, terminal, samples_data = self.partial_rollout(
                env=worker_env,
                policy=self.policy,
                t=t,
                batch_size=self.batch_size,
                max_path_length=self.max_path_length,
                last_obs=last_obs,
                run_event=run_event
            )

            self.worker_perform_update(samples_data, last_obs, terminal)

            path_samples_data.append(samples_data)

            if terminal or t >= self.max_path_length:
                # aggregate to a single trajectory
                path = dict(
                    observations=tensor_utils.concat_tensor_list([x["observations"] for x in path_samples_data]),
                    actions=tensor_utils.concat_tensor_list([x["actions"] for x in path_samples_data]),
                    rewards=tensor_utils.concat_tensor_list([x["rewards"] for x in path_samples_data]),
                    agent_infos=tensor_utils.concat_tensor_dict_list([x["agent_infos"] for x in path_samples_data]),
                )
                self.worker_record_statistics(worker_id, path)
                # should start a new trajectory
                last_obs = None
                self.T += t
                t = 0
                path_samples_data = []

    def worker_record_statistics(self, worker_id, path):
        queue = self.worker_stats_queue
        undiscounted_return = np.sum(path["rewards"])
        agent_infos = path["agent_infos"]
        ent = np.mean(self.policy.distribution.entropy(agent_infos))
        queue.put([
            ("AverageReturn", np.mean, undiscounted_return),
            ("AverageDiscountedReturn", np.mean, special.discount_cumsum(path["rewards"], self.discount)[0]),
            ("NumTrajs", np.sum, 1),
            ("StdReturn", np.std, undiscounted_return),
            ("MaxReturn", np.max, undiscounted_return),
            ("MinReturn", np.min, undiscounted_return),
            ("Entropy", np.mean, ent),
            ("Perplexity", aggregate_perplexity, ent),
            # ("AvgLogStd", np.mean, np.mean(agent_infos["log_std"])),
            # ("AvgStd", np.mean, np.mean(np.exp(agent_infos["log_std"]))),
        ])

    def partial_rollout(self, env, policy, t, batch_size, max_path_length, last_obs, run_event):
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        terminal = False
        if last_obs is None:
            obs = env.reset()
        else:
            obs = last_obs
        while run_event.is_set() and not terminal and t < max_path_length and len(observations) < batch_size:
            action, agent_info = policy.get_action(obs)
            next_obs, reward, terminal, env_info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            agent_infos.append(agent_info)
            obs = next_obs
            t += 1
        return obs, t, terminal, dict(
            observations=tensor_utils.stack_tensor_list(env.observation_space.flatten_n(observations)),
            actions=tensor_utils.stack_tensor_list(env.action_space.flatten_n(actions)),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        )

    def worker_perform_update(self, samples_data, last_obs, terminal):
        # train the policy and critic

        # first compute the advantages
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        rewards = samples_data["rewards"]
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]

        f_train_policy = self.opt_info["f_train_policy"]
        f_train_critic = self.opt_info["f_train_critic"]

        if terminal:
            cum_return = 0
        else:
            cum_return = self.critic_network.predict([last_obs])[0][0]
        predicted = self.critic_network.predict(observations)
        returns = []
        for reward, predicted_v in zip(rewards[::-1], predicted[::-1]):
            cum_return = self.discount * cum_return + reward * self.scale_reward
            returns.append(cum_return)

        returns = np.asarray(returns, dtype=np.float32)

        all_inputs = [observations, actions, returns] + state_info_list

        f_train_policy(*all_inputs)
        f_train_critic(*all_inputs)

    def train(self):
        self.init_opt()
        run_event = threading.Event()
        run_event.set()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.worker_envs = []
            workers = []
            for worker_id in range(self.n_workers):
                self.worker_envs.append(pickle.loads(pickle.dumps(self.env)))
                # self.worker_stats[worker_id] = dict()
                t = threading.Thread(target=run_algo, args=(self, worker_id, sess, run_event))
                t.start()
                workers.append(t)
            last_T = self.T
            try:
                while True:
                    time.sleep(1)
                    if self.T - last_T >= self.epoch_length:
                        all_stats = []
                        # get all items from queue
                        while True:
                            try:
                                all_stats.append(self.worker_stats_queue.get_nowait())
                            except queue.Empty as e:
                                break
                        # log new results
                        epoch = self.T / self.epoch_length
                        logger.record_tabular("Epoch", epoch)
                        logger.record_tabular("T", self.T)
                        kvs = dict()
                        for stats in all_stats:
                            for k, op, v in stats:
                                if (k, op) not in kvs:
                                    kvs[(k, op)] = list()
                                kvs[(k, op)].append(v)
                        for (k, op), vals in kvs.items():
                            logger.record_tabular(k, op(vals))
                        logger.dump_tabular()
                        last_T = self.T
            except Exception:
                print("Exception here")
                run_event.clear()
                for worker in workers:
                    worker.join(10)
                raise
