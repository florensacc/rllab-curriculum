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
import lasagne.updates
import theano.tensor as TT
import numpy as np
import cPickle as pickle
import time
import contextlib


def start_worker(algo, worker_id, shared_T, pipe):
    ext.set_seed(ext.get_seed() + worker_id)
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


@contextlib.contextmanager
def using_local_memory(*param_objs):
    """
    Upon entering the context, replace all parameters for the provided parameterized objects to using local memory,
    so that they will remain fixed during the whole body. This is useful to make gradient computation more stable.
    :param param_objs:
    :return:
    """
    shared_params = []
    # replace to local parameters
    for param_obj in param_objs:
        obj_shared_params = []
        for param in param_obj.get_params():
            param_val = param.get_value(borrow=True)
            obj_shared_params.append(param_val)
            param.set_value(np.copy(param_val), borrow=True)
        shared_params.append(obj_shared_params)
    yield
    for param_obj, obj_shared_params in zip(param_objs, shared_params):
        for param, shared_param in zip(param_obj.get_params(), obj_shared_params):
            param.set_value(shared_param, borrow=True)


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


class Diagnostics(object):
    def __init__(self):
        manager = mp.Manager()
        self.lock = manager.RLock()
        self.sample_time = mp.Value('f', 0.)
        self.update_time = mp.Value('f', 0.)
        self.es_path_returns = manager.list()
        self.qf_loss_averages = manager.list()
        self.policy_surr_averages = manager.list()
        self.q_averages = manager.list()
        self.y_averages = manager.list()


def new_shared_diagnostics():
    return Diagnostics()


def apply_adam_update(param_obj, grads, adam_state, learning_rate):
    beta1 = adam_state.beta1
    beta2 = adam_state.beta2
    # with adam_state.t.get_lock():
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
        param_val = param.get_value(borrow=True)
        target_val = target_param.get_value(borrow=True)
        np.copyto(target_val, target_val * (1. - soft_target_tau) + param_val * soft_target_tau)


def apply_weight_decay(param_obj, learning_rate, weight_decay):
    for param in param_obj.get_params():
        param_val = param.get_value(borrow=True)
        np.copyto(param_val, param_val * (1.0 - learning_rate * weight_decay))


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
            es=None,
            worker_es=None,
            manager=None,
            target_policy=None,
            target_qf=None,
            policy_adam_state=None,
            qf_adam_state=None,
            shared_diagnostics=None,
            n_workers=1,
            discount=0.99,
            scale_reward=1.,
            max_path_length=100,
            use_replay_pool=True,
            min_eval_interval=10000,
            min_pool_size=10000,
            max_pool_size=1000000,
            max_samples=25000000,
            evaluate_policy=True,
            eval_samples=10000,
            target_update_method='soft',
            hard_target_interval=40000,
            soft_target_tau=1e-3,
            qf_weight_decay=0.,
            # qf_update_method='adam',
            qf_learning_rate=1e-3,
            policy_weight_decay=0,
            # policy_update_method='adam',
            policy_learning_rate=1e-4,
            batch_size=32,
            debug=False,
            sync_mode='all'
    ):
        """
        :param sync_mode: Can be one of 'all', 'none'.
            all: Synchronize everything. This is the slowest version. Only one process will be allowed to compute the
                gradient at a given time.
            none: Totally asynchronous.
        :return:
        """
        assert sync_mode in ['all', 'none']
        if es is None and worker_es is None:
            raise ValueError("Must provide at least one of `es` and `worker_es` parameters")
        if worker_es is not None and len(worker_es) != n_workers:
            raise ValueError("The size of the `worker_es` list parameter must be equal to `n_workers`")
        # Make sure all variables are properly initialized when serialized to worker copies
        if manager is None:
            manager = mp.Manager()
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
        if shared_diagnostics is None:
            shared_diagnostics = new_shared_diagnostics()
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.qf = qf
        self.target_policy = target_policy
        self.target_qf = target_qf
        self.es = es
        self.worker_es = worker_es
        self.n_workers = n_workers
        self.discount = discount
        self.scale_reward = scale_reward
        self.sync_mode = sync_mode

        self.max_path_length = max_path_length
        self.use_replay_pool = use_replay_pool

        self.min_eval_interval = min_eval_interval
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.max_samples = max_samples
        self.evaluate_policy = evaluate_policy
        self.eval_samples = eval_samples
        self.target_update_method = target_update_method
        self.hard_target_interval = hard_target_interval
        self.soft_target_tau = soft_target_tau

        self.qf_weight_decay = qf_weight_decay
        self.qf_learning_rate = qf_learning_rate
        self.qf_adam_state = qf_adam_state
        self.policy_weight_decay = policy_weight_decay
        self.policy_learning_rate = policy_learning_rate
        self.policy_adam_state = policy_adam_state
        self.shared_diagnostics = shared_diagnostics
        self.debug = debug

        # self.qf_loss_averages = qf_loss_averages
        # self.policy_surr_averages = policy_surr_averages
        # self.q_averages = q_averages
        # self.y_averages = y_averages

        self.batch_size = batch_size

        self.f_policy_grads = None
        self.f_qf_grads = None
        self._start_time = time.time()

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
                finished = shared_T.value >= self.max_samples
                if shared_T.value >= last_eval_T + self.min_eval_interval or finished:
                    last_eval_T = shared_T.value
                    if self.evaluate_policy:
                        eval_policy.set_param_values(self.policy.get_param_values())
                        self.evaluate(eval_env, eval_policy, last_eval_T)
                    if finished:
                        break
                time.sleep(0.01)
            print("finished. joining...")
            for p in processes:
                p.terminate()
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
        policy_reg_param_norm = np.linalg.norm(
            self.policy.get_param_values(regularizable=True)
        )
        qfun_reg_param_norm = np.linalg.norm(
            self.qf.get_param_values(regularizable=True)
        )

        if self.debug:
            with self.shared_diagnostics.lock:

                sample_time = self.shared_diagnostics.sample_time.value
                update_time = self.shared_diagnostics.update_time.value
                self.shared_diagnostics.sample_time.value = 0
                self.shared_diagnostics.update_time.value = 0

                if len(self.shared_diagnostics.q_averages) > 0:
                    all_qs = np.concatenate(self.shared_diagnostics.q_averages)
                else:
                    all_qs = []
                if len(self.shared_diagnostics.y_averages) > 0:
                    all_ys = np.concatenate(self.shared_diagnostics.y_averages)
                else:
                    all_ys = []

                average_q_loss = np.mean(self.shared_diagnostics.qf_loss_averages)
                average_policy_surr = np.mean(self.shared_diagnostics.policy_surr_averages)

                del self.shared_diagnostics.qf_loss_averages[:]
                del self.shared_diagnostics.policy_surr_averages[:]

                del self.shared_diagnostics.q_averages[:]
                del self.shared_diagnostics.y_averages[:]

        else:
            sample_time = self.shared_diagnostics.sample_time.value
            update_time = self.shared_diagnostics.update_time.value
            self.shared_diagnostics.sample_time.value = 0
            self.shared_diagnostics.update_time.value = 0

        logger.record_tabular('NSamples', T)
        logger.record_tabular('TimeSinceStart', time.time() - self._start_time)
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
        if self.debug:
            logger.record_tabular('AverageQLoss', average_q_loss)
            logger.record_tabular('AveragePolicySurr', average_policy_surr)
            if len(all_qs) > 0:
                logger.record_tabular('AverageQ', np.mean(all_qs))
                logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
                logger.record_tabular('AverageY', np.mean(all_ys))
                logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
                logger.record_tabular('AverageAbsQYDiff',
                                      np.mean(np.abs(all_qs - all_ys)))
        logger.record_tabular('AverageAction', average_action)
        logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)
        logger.record_tabular('QFunRegParamNorm',
                              qfun_reg_param_norm)
        logger.record_tabular('SampleTime', sample_time)
        logger.record_tabular('UpdateTime', update_time)
        eval_env.log_diagnostics(paths)
        eval_policy.log_diagnostics(paths)
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

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = TT.mean(TT.square(yvar - qval))

        policy_qval = self.qf.get_qval_sym(
            obs, self.policy.get_action_sym(obs),
            deterministic=True
        )
        policy_surr = -TT.mean(policy_qval)

        qf_grads = TT.grad(qf_loss, self.qf.get_params(trainable=True))
        policy_grads = TT.grad(policy_surr, self.policy.get_params(trainable=True))

        self.f_qf_grads = ext.compile_function(
            inputs=[yvar, obs, action],
            outputs=[qf_loss, qval] + list(qf_grads),
        )

        self.f_policy_grads = ext.compile_function(
            inputs=[obs],
            outputs=[policy_surr] + list(policy_grads),
        )

    def worker_train(self, worker_id, shared_T, pipe):
        """
        Training procedure on worker processes
        :param worker_id: ID assigned to the worker for logging purposes
        :param shared_T: a shared counter for the global time step

        """

        # Each worker needs to compile their own functions
        logger.push_prefix("[Worker %d] | " % worker_id)
        # logger.log("Initializing")
        self.worker_init_opt()
        terminal = False
        obs = self.env.reset()
        t = 0

        if self.worker_es is not None:
            es = self.worker_es[worker_id]
        else:
            es = self.es

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
                es.reset()

            # increment global counter
            current_shared_T = None
            with shared_T.get_lock():
                if shared_T.value >= self.max_samples:
                    return
                shared_T.value += 1
                current_shared_T = shared_T.value

            if self.target_update_method == 'hard' and current_shared_T % self.hard_target_interval == 0:
                apply_target_update(self.qf, self.target_qf, 1.)
                apply_target_update(self.policy, self.target_policy, 1.)

            start_time = time.time()
            action = es.get_action(current_shared_T, obs, self.policy)
            next_obs, reward, terminal, _ = self.env.step(action)
            t += 1

            sample_time = time.time() - start_time
            self.shared_diagnostics.sample_time.value += sample_time

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

                #logger.log("training")
                start_time = time.time()

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

                if self.sync_mode == 'all':
                    with self.qf_adam_state.t.get_lock():

                        next_actions, _ = self.target_policy.get_actions(next_observations)
                        next_qvals = self.target_qf.get_qval(next_observations, next_actions)
                        ys = rewards + (1. - terminals) * self.discount * next_qvals

                        results = self.f_qf_grads(ys, observations, actions)
                        qf_loss, qval = results[:2]
                        qf_grads = results[2:]
                        results = self.f_policy_grads(observations)
                        policy_surr = results[0]
                        policy_grads = results[1:]

                        apply_adam_update(self.qf, qf_grads, self.qf_adam_state, learning_rate=self.qf_learning_rate)
                        apply_adam_update(self.policy, policy_grads, self.policy_adam_state,
                                          learning_rate=self.policy_learning_rate)

                        apply_weight_decay(self.qf, learning_rate=self.qf_learning_rate,
                                           weight_decay=self.qf_weight_decay)
                        apply_weight_decay(self.policy, learning_rate=self.policy_learning_rate,
                                           weight_decay=self.policy_weight_decay)

                        apply_target_update(self.qf, self.target_qf, self.soft_target_tau)
                        apply_target_update(self.policy, self.target_policy, self.soft_target_tau)
                else:
                    with using_local_memory(self.target_qf, self.target_policy, self.policy, self.qf):
                        next_actions, _ = self.target_policy.get_actions(next_observations)
                        next_qvals = self.target_qf.get_qval(next_observations, next_actions)
                        ys = rewards + (1. - terminals) * self.discount * next_qvals

                        results = self.f_qf_grads(ys, observations, actions)
                        qf_loss, qval = results[:2]
                        qf_grads = results[2:]
                        results = self.f_policy_grads(observations)
                        policy_surr = results[0]
                        policy_grads = results[1:]

                    apply_adam_update(self.qf, qf_grads, self.qf_adam_state, learning_rate=self.qf_learning_rate)
                    apply_adam_update(self.policy, policy_grads, self.policy_adam_state,
                                      learning_rate=self.policy_learning_rate)

                    apply_weight_decay(self.qf, learning_rate=self.qf_learning_rate, weight_decay=self.qf_weight_decay)
                    apply_weight_decay(self.policy, learning_rate=self.policy_learning_rate,
                                       weight_decay=self.policy_weight_decay)

                    if self.target_update_method == 'soft':
                        apply_target_update(self.qf, self.target_qf, self.soft_target_tau)
                        apply_target_update(self.policy, self.target_policy, self.soft_target_tau)

                update_time = time.time() - start_time

                if self.debug:
                    with self.shared_diagnostics.lock:
                        self.shared_diagnostics.qf_loss_averages.append(qf_loss)
                        self.shared_diagnostics.policy_surr_averages.append(policy_surr)
                        self.shared_diagnostics.q_averages.append(qval)
                        self.shared_diagnostics.y_averages.append(ys)
                        self.shared_diagnostics.update_time.value += update_time
                else:
                    self.shared_diagnostics.update_time.value += update_time

                if not self.use_replay_pool:
                    observations = []
                    next_observations = []
                    actions = []
                    rewards = []
                    terminals = []
                #logger.log("trained")
