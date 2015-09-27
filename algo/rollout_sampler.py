from joblib.pool import MemmapingPool
from contextlib import contextmanager
from joblib.parallel import SafeFunction
import numpy as np
import os
import theano
import theano.tensor as T
import theano.sandbox.cuda
from collections import defaultdict
from misc.logging import Message, log, prefix_log
from joblib import Parallel, delayed

@contextmanager
def ensure_cpu():
    default_device = theano.sandbox.cuda.use.device_number
    if default_device is not None:
        theano.sandbox.cuda.unuse()
    yield
    if default_device is not None:
        theano.sandbox.cuda.use('gpu%d' % default_device, force=True)

def _init_subprocess(gen_mdp, gen_policy):
    global mdp
    global policy
    np.random.seed(os.getpid())
    mdp, policy = _init_mdp_policy(gen_mdp, gen_policy)

def _init_mdp_policy(gen_mdp, gen_policy):
    mdp = gen_mdp()
    input_var = T.matrix('input') # N*Ds
    policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
    return mdp, policy

def _collect_samples(mdp, policy, itr, param_values, max_samples, max_steps, discount):#, args):
    total_q_vals = defaultdict(int)
    action_visits = defaultdict(int)
    traj = []
    samples = []
    tot_rewards = 0
    n_traj = 0

    log('starting...')
    policy.set_param_values(param_values)

    last_displayed = 0
    n_samples = 0
    n_steps = 0

    state, obs = mdp.sample_initial_state()

    while n_samples < max_samples and n_steps < max_steps:
        if not np.isinf(max_steps) and n_steps / 1000 > last_displayed:
            log('%d / %d steps (%d samples; %d traj)' % (n_steps, max_steps, n_samples, n_traj))
            last_displayed += 1
        elif not np.isinf(max_samples) and n_samples / 1000 > last_displayed:
            log('%d / %d samples (%d steps; %d traj)' % (n_samples, max_samples, n_steps, n_traj))
            last_displayed += 1

        actions, action_probs = policy.get_actions_single(obs)
        next_state, next_obs, reward, done, steps = mdp.step_single(state, actions)
        n_steps += steps
        n_samples += 1
        traj.append((obs, actions, next_obs, reward))
        samples.append((obs, actions, action_probs))
        tot_rewards += reward
        if done or n_samples >= max_samples or n_steps >= max_steps:#effective_steps >= n_samples:
            n_traj += 1
            # update all Q-values along this trajectory
            cum_reward = 0
            for obs, actions, next_obs, reward in traj[::-1]:
                cum_reward = discount * cum_reward + reward
                action_pair = (tuple(obs), tuple(actions))
                total_q_vals[action_pair] += cum_reward
                action_visits[action_pair] += 1
            traj = []
        state, obs = next_state, next_obs

    N = len(samples)

    all_obs = np.zeros((N,) + mdp.observation_shape)
    Q_est = np.zeros(N)
    all_pi_old = [np.zeros((N, Da)) for Da in mdp.action_dims]
    all_actions = [np.zeros(N, dtype='uint8') for _ in mdp.action_dims]
    for idx, tpl in enumerate(samples):
        obs, actions, action_probs = tpl
        for ia, action in enumerate(actions):
            all_actions[ia][idx] = action
        for ia, probs in enumerate(action_probs):
            all_pi_old[ia][idx,:] = probs
        action_pair = (tuple(obs), tuple(actions))
        Q_est[idx] = total_q_vals[action_pair] / action_visits[action_pair]
        all_obs[idx] = obs

    return tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions

def _subprocess_collect_samples(itr, param_values, max_samples, max_steps, discount):
    global mdp
    global policy
    return _collect_samples(mdp, policy, itr, param_values, max_samples, max_steps, discount)

def _combine_samples(results):
    rewards_list, n_traj_list, all_obs_list, Q_est_list, all_pi_old_list, all_actions_list = map(list, zip(*results))
    tot_rewards = sum(rewards_list)
    n_traj = sum(n_traj_list)
    all_obs = np.concatenate(all_obs_list)
    Q_est = np.concatenate(Q_est_list)
    na = len(all_pi_old_list[0])
    all_pi_old = [np.concatenate(map(lambda x: x[i], all_pi_old_list)) for i in range(na)]
    all_actions = [np.concatenate(map(lambda x: x[i], all_actions_list)) for i in range(na)]
    return tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions

class RolloutSampler(object):

    def __init__(self, n_parallel, gen_mdp, gen_policy):
        self._n_parallel = n_parallel
        self._setup_called = False
        self._gen_mdp = gen_mdp
        self._gen_policy = gen_policy

    def _setup(self):
        if not self._setup_called:
            if self._n_parallel > 1:
                with ensure_cpu():
                    self._parallel = Parallel(n_jobs=self._n_parallel)#= pool = MemmapingPool(n_parallel)
                    self._parallel.__enter__()
                    self._parallel(delayed(_init_subprocess)(self._gen_mdp, self._gen_policy) \
                            for _ in range(self._n_parallel))
            else:
                self._mdp, self._policy = _init_mdp_policy(self._gen_mdp, self._gen_policy)
            self._setup_called = True

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._n_parallel > 1:
            self._parallel.__exit__(exc_type, exc_value, traceback)
        self._setup_called = False

    def collect_samples(self, itr, param_values, max_samples, max_steps, discount):
        if not self._setup_called:
            raise ValueError('Must enclose RolloutSampler in a with clause')
        if self._n_parallel > 1:
            with ensure_cpu():
                result_list = self._parallel(delayed(_subprocess_collect_samples)(
                    itr, param_values, max_samples / self._n_parallel, max_steps / self._n_parallel, discount)
                        for _ in range(self._n_parallel))
                return _combine_samples(result_list)
        else:
            return _collect_samples(self._mdp, self._policy, itr, param_values, max_samples, max_steps, discount)
