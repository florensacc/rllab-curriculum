from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from rllab.misc import logger
from sandbox.rocky.hogwild.shared_parameterized import SharedParameterized
from functools import partial
from rllab.misc import ext
from collections import namedtuple
from rllab.sampler.parallel_sampler import rollout
from rllab.misc import special
from sandbox.rocky.hogwild.shared_parameterized import new_shared_mem_array
import multiprocessing as mp
import lasagne
import theano.tensor as TT
import numpy as np
import cPickle as pickle
import time


def start_worker(algo, worker_id, shared_T, pipe):
    algo.worker_train(worker_id=worker_id, shared_T=shared_T, pipe=pipe)


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'sgd':
        return partial(lasagne.updates.sgd, **ext.compact(kwargs))
    else:
        raise NotImplementedError


def new_params_like(param_obj):
    new_params = []
    for param in param_obj.get_params(trainable=True):
        new_params.append(new_shared_mem_array(np.zeros_like(param.get_value(borrow=True))))
    return new_params


AdamState = namedtuple('AdamState', ['t', 'm_t', 'v_t', 'beta1', 'beta2', 'epsilon'])


def new_adam_state(param_obj):
    m_t = new_params_like(param_obj)
    v_t = new_params_like(param_obj)
    t = mp.Value('i', 0)
    return AdamState(
        t=t,
        m_t=m_t,
        v_t=v_t,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )


def apply_adam_update(param_obj, grads, adam_state, learning_rate):
    beta1 = adam_state.beta1
    beta2 = adam_state.beta2
    with adam_state.t.get_lock():
        adam_state.t.value += 1
        t_new = adam_state.t.value
        a_new = learning_rate * np.sqrt(1 - beta2 ** t_new) / (1 - beta1 ** t_new)
        for param, g_t, m_t, v_t in zip(param_obj.get_params(trainable=True), grads, adam_state.m_t, adam_state.v_t):
            param_val = param.get_value(borrow=True)
            m_new = beta1 * m_t + (1 - beta1) * g_t
            v_new = beta2 * v_t + (1 - beta2) * g_t ** 2
            step = a_new * m_new / (np.sqrt(v_new) + adam_state.epsilon)

            np.copyto(m_t, m_new)
            np.copyto(v_t, v_new)
            np.copyto(param_val, param_val - step)

            # adam_state.m_t = m_t
            # adam_state.v_t = v_t
            # adam_state.t = t


def apply_target_update(param_obj, target_param_obj, soft_target_tau):
    for param, target_param in zip(param_obj.get_params(), target_param_obj.get_params()):
        target_val = target_param.get_value(borrow=True)
        param_val = param.get_value(borrow=True)
        np.copyto(target_val, target_val * (1 - soft_target_tau) + param_val * soft_target_tau)


class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size


class AsyncDDPG(RLAlgorithm, Serializable):
    """
    Asynchronous Deep Deterministic Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            qf,
            es,
            target_policy=None,
            target_qf=None,
            policy_adam_state=None,
            qf_adam_state=None,
            n_workers=1,
            discount=0.99,
            scale_reward=1.,
            max_path_length=100,
            use_replay_pool=True,
            min_eval_interval=10000,
            min_pool_size=10000,
            max_pool_size=1000000,
            max_samples=25000000,
            eval_samples=10000,
            soft_target_tau=1e-3,
            qf_weight_decay=0.,
            # qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            # policy_update_method='adam',
            policy_learning_rate=1e-4,
            batch_size=32,
    ):
        # Make sure all variables are properly initialized when serialized to worker copies
        if not isinstance(policy, SharedParameterized):
            policy = SharedParameterized(policy)
        if not isinstance(qf, SharedParameterized):
            qf = SharedParameterized(qf)
        if target_policy is None:
            target_policy = policy.new_mem_copy()
        if target_qf is None:
            target_qf = qf.new_mem_copy()
        if policy_adam_state is None:
            policy_adam_state = new_adam_state(policy)
        if qf_adam_state is None:
            qf_adam_state = new_adam_state(qf)
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_policy = target_policy
        self.target_qf = target_qf
        self.es = es
        self.n_workers = n_workers
        self.discount = discount
        self.scale_reward = scale_reward

        self.max_path_length = max_path_length
        self.use_replay_pool = use_replay_pool
        self.min_eval_interval = min_eval_interval
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.max_samples = max_samples
        self.eval_samples = eval_samples
        self.soft_target_tau = soft_target_tau

        self.qf_weight_decay = qf_weight_decay
        self.qf_learning_rate = qf_learning_rate
        self.qf_adam_state = qf_adam_state
        self.policy_weight_decay = policy_weight_decay
        self.policy_learning_rate = policy_learning_rate
        self.policy_adam_state = policy_adam_state

        self.batch_size = batch_size

        self.f_policy_grads = None
        self.f_qf_grads = None

    def train(self):
        """
        Start the training procedure on the master process. It launches several worker processes.
        """
        eval_policy = self.policy.new_mem_copy()
        eval_env = pickle.loads(pickle.dumps(self.env))

        processes = []
        pipes = []
        shared_T = mp.Value('i', 0)
        # if self.n_workers == 1:
        #     start_worker(self, 0, shared_T, pipe)
        # else:
        for id in xrange(self.n_workers):
            pipe = mp.Pipe()
            p = mp.Process(target=start_worker, args=(self, id, shared_T, pipe))
            p.start()
            processes.append(p)
            pipes.append(pipe)
        last_eval_T = 0
        try:
            while True:
                # for p, pipe in zip(processes, pipes):
                #     parent_conn = pipe[0]
                #     parent_conn.
                if shared_T.value >= last_eval_T + self.min_eval_interval:
                    last_eval_T = shared_T.value
                    eval_policy.set_param_values(self.policy.get_param_values())
                    self.evaluate(eval_env, eval_policy, last_eval_T)
                time.sleep(0.01)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()

    def evaluate(self, eval_env, eval_policy, T):
        paths = []
        samples_count = 0
        while samples_count < self.eval_samples:
            path = rollout(eval_env, eval_policy, self.max_path_length)
            paths.append(path)
            samples_count += len(path["rewards"])

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )
        returns = [sum(path["rewards"]) for path in paths]
        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))
        logger.record_tabular('NSamples', T)
        logger.record_tabular('AverageReturn',
                              np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageAction', average_action)
        logger.dump_tabular()

    def worker_init_opt(self):
        """
        Initialize the computation graph on the worker process
        """
        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = TT.vector('ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([TT.sum(TT.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([TT.sum(TT.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -TT.mean(policy_qval)

        policy_reg_surr = policy_surr + policy_weight_decay_term

        qf_grads = TT.grad(qf_reg_loss, self.qf.get_params(trainable=True))
        policy_grads = TT.grad(policy_reg_surr, self.policy.get_params(trainable=True))

        self.f_qf_grads = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=qf_grads,
        )

        self.f_policy_grads = ext.compile_function(
            inputs=[obs],
            outputs=policy_grads,
        )

    def worker_train(self, worker_id, shared_T, pipe):
        """
        Training procedure on worker processes
        :param worker_id: ID assigned to the worker for logging purposes
        :param shared_T: a shared counter for the global time step
        :param pipe: a multiprocessing.Pipe object for communicating with the master process
        """

        # Each worker needs to compile their own functions
        logger.push_prefix("[Worker %d] | " % worker_id)
        # logger.log("Initializing")
        self.worker_init_opt()
        terminal = True
        obs = None
        t = 0

        if self.use_replay_pool:
            pool = SimpleReplayPool(
                max_pool_size=self.max_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_dim
            )
        else:
            observations = []
            next_observations = []
            actions = []
            rewards = []
            terminals = []
        while True:
            if terminal:
                obs = self.env.reset()
                t = 0
                self.es.reset()

            t += 1
            # increment global counter
            current_shared_T = None
            with shared_T.get_lock():
                shared_T.value += 1
                current_shared_T = shared_T.value
            # if current_shared_T %

            action = self.es.get_action(current_shared_T, obs, self.policy)
            next_obs, reward, terminal, _ = self.env.step(action)

            if not terminal and t >= self.max_path_length:
                terminal = True
            else:
                if self.use_replay_pool:
                    pool.add_sample(obs, action, reward * self.scale_reward, terminal)
                else:
                    observations.append(self.env.observation_space.flatten(obs))
                    next_observations.append(self.env.observation_space.flatten(next_obs))
                    actions.append(self.env.action_space.flatten(action))
                    rewards.append(reward * self.scale_reward)
                    terminals.append(terminal)

            obs = next_obs

            if self.use_replay_pool and pool.size >= self.min_pool_size or \
                            not self.use_replay_pool and len(observations) >= self.batch_size:

                if self.use_replay_pool:
                    batch = pool.random_batch(self.batch_size)
                    observations = batch["observations"]
                    next_observations = batch["next_observations"]
                    rewards = batch["rewards"]
                    actions = batch["actions"]
                    terminals = batch["terminals"]
                else:
                    observations = np.array(observations)
                    next_observations = np.array(next_observations)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    terminals = np.array(terminals)

                next_actions, _ = self.target_policy.get_actions(next_observations)
                next_qvals = self.target_qf.get_qval(next_observations, next_actions)
                ys = rewards + (1. - terminals) * self.discount * next_qvals
                qf_grads = self.f_qf_grads(ys, observations, actions)
                policy_grads = self.f_policy_grads(observations)

                apply_adam_update(self.qf, qf_grads, self.qf_adam_state, learning_rate=self.qf_learning_rate)
                apply_adam_update(self.policy, policy_grads, self.policy_adam_state,
                                  learning_rate=self.policy_learning_rate)
                apply_target_update(self.qf, self.target_qf, self.soft_target_tau)# / self.n_workers)
                apply_target_update(self.policy, self.target_policy, self.soft_target_tau)# / self.n_workers)

                if not self.use_replay_pool:
                    observations = []
                    next_observations = []
                    actions = []
                    rewards = []
                    terminals = []
