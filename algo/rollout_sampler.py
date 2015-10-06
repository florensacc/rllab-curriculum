from joblib.pool import MemmapingPool
from joblib.parallel import SafeFunction
import numpy as np
import os
import theano.tensor as T
import pyprind
from collections import defaultdict
from misc.console import log
from multiprocessing import Manager
from sampler import launch_sampler
import pickle


def _init_subprocess(*args):
    if len(args) == 1:
        _, gen_mdp, gen_policy, _ = pickle.loads(args[0])
    else:
        gen_mdp, gen_policy = args
    global mdp
    global policy
    np.random.seed(os.getpid())
    mdp, policy = _init_mdp_policy(gen_mdp, gen_policy)


def _init_mdp_policy(gen_mdp, gen_policy):
    mdp = gen_mdp()
    input_var = T.matrix('input')  # N*Ds
    policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
    return mdp, policy


def _subprocess_collect_samples(args):
    itr, param_values, max_samples, max_steps, discount, queue = args
    global mdp
    global policy
    return _collect_samples(mdp, policy, itr, param_values, max_samples, max_steps, discount, queue)


def _collect_samples(mdp, policy, itr, param_values, max_samples, max_steps, discount, queue=None):
    try:
        total_q_vals = defaultdict(int)
        action_visits = defaultdict(int)
        traj = []
        samples = []
        tot_rewards = 0
        n_traj = 0

        policy.set_param_values(param_values)

        last_displayed = 0
        last_n_steps = 0
        last_n_samples = 0
        n_samples = 0
        n_steps = 0

        state, obs = mdp.sample_initial_state()

        while n_samples < max_samples and n_steps < max_steps:
            if not np.isinf(max_steps) and n_steps / 100 > last_displayed:
                last_displayed += 1
                if queue is not None:
                    queue.put(('steps', n_steps - last_n_steps))
                    last_n_steps = n_steps
            elif not np.isinf(max_samples) and n_samples / 100 > last_displayed:
                last_displayed += 1
                if queue is not None:
                    queue.put(('samples', n_samples - last_n_samples))
                    last_n_samples = n_samples
            actions, action_probs = policy.get_actions_single(obs)
            next_state, next_obs, reward, done, steps = mdp.step_single(state, actions)
            if reward != 0:
                print reward
            n_steps += steps
            n_samples += 1
            traj.append((state, obs, actions, next_obs, reward))
            samples.append((state, obs, actions, action_probs))
            tot_rewards += reward
            if done or n_samples >= max_samples or n_steps >= max_steps:
                n_traj += 1
                # update all Q-values along this trajectory
                cum_reward = 0
                for state, obs, actions, next_obs, reward in traj[::-1]:
                    cum_reward = discount * cum_reward + reward
                    action_pair = (tuple(obs), tuple(actions))
                    total_q_vals[action_pair] += cum_reward
                    action_visits[action_pair] += 1
                traj = []
            state, obs = next_state, next_obs

        N = len(samples)

        all_obs = np.zeros((N,) + policy.observation_shape)
        all_states = np.zeros((N,) + samples[0][0].shape)
        Q_est = np.zeros(N)
        all_pi_old = [np.zeros((N, Da)) for Da in policy.action_dims]
        all_actions = [np.zeros(N, dtype='uint8') for _ in policy.action_dims]
        for idx, tpl in enumerate(samples):
            state, obs, actions, action_probs = tpl
            for ia, action in enumerate(actions):
                all_actions[ia][idx] = action
            for ia, probs in enumerate(action_probs):
                all_pi_old[ia][idx,:] = probs
            action_pair = (tuple(obs), tuple(actions))
            Q_est[idx] = total_q_vals[action_pair] / action_visits[action_pair]
            all_obs[idx] = obs
            all_states[idx] = state

        return tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions, all_states
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

def _combine_samples(results):
    rewards_list, n_traj_list, all_obs_list, Q_est_list, all_pi_old_list, all_actions_list, all_states_list = map(list, zip(*results))
    tot_rewards = sum(rewards_list)
    n_traj = sum(n_traj_list)
    all_obs = np.concatenate(all_obs_list)
    Q_est = np.concatenate(Q_est_list)
    na = len(all_pi_old_list[0])
    all_pi_old = [np.concatenate(map(lambda x: x[i], all_pi_old_list)) for i in range(na)]
    all_actions = [np.concatenate(map(lambda x: x[i], all_actions_list)) for i in range(na)]
    all_states = np.concatenate(all_states_list)
    return tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions, all_states

class RolloutSampler(object):

    def __init__(self, buf=None, n_parallel=None, gen_mdp=None, gen_policy=None):
        if buf is not None:
            n_parallel, gen_mdp, gen_policy, _ = pickle.loads(buf)
        self._n_parallel = n_parallel
        self._setup_called = False
        self._gen_mdp = gen_mdp
        self._gen_policy = gen_policy
        self._pool = None
        self._mdp = None
        self._policy = None
        self._buf = buf

    def _setup(self):
        if not self._setup_called:
            if self._n_parallel > 1:
                self._pool = MemmapingPool(self._n_parallel)
                if self._buf is not None:
                    self._pool.map(SafeFunction(_init_subprocess), [self._buf] * self._n_parallel)
                else:
                    self._pool.map(SafeFunction(_init_subprocess), [(self._gen_mdp, self._gen_policy)] * self._n_parallel)
            else:
                self._mdp, self._policy = _init_mdp_policy(self._gen_mdp, self._gen_policy)
            self._setup_called = True

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._n_parallel > 1:
            self._pool.close()
            self._pool.terminate()
            self._pool = None
        else:
            self._mdp = None
            self._policy = None
        self._setup_called = False

    def collect_samples(self, itr, param_values, max_samples, max_steps, discount):
        if not self._setup_called:
            raise ValueError('Must enclose RolloutSampler in a with clause')
        if self._n_parallel > 1:
            manager = Manager()
            queue = manager.Queue()
            args = itr, param_values, max_samples / self._n_parallel, max_steps / self._n_parallel, discount, queue
            map_result = self._pool.map_async(_subprocess_collect_samples, [args] * self._n_parallel)
            n_samples = 0
            n_steps = 0
            max_progress = 1000000
            cur_progress = 0
            pbar = pyprind.ProgBar(max_progress)
            while not map_result.ready():
                map_result.wait(0.1)
                while not queue.empty():
                    ret = queue.get_nowait()
                    if ret[0] == 'steps':
                        n_steps += ret[1]
                    elif ret[0] == 'samples':
                        n_samples += ret[1]
                new_progress = max(n_samples * max_progress / max_samples, n_steps * max_progress / max_steps)
                pbar.update(new_progress - cur_progress)
                cur_progress = new_progress
            pbar.stop()
            try:
                result_list = map_result.get()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
            return _combine_samples(result_list)
        else:
            return _collect_samples(self._mdp, self._policy, itr, param_values, max_samples, max_steps, discount)

if __name__ == '__main__':
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    launch_sampler(RolloutSampler)
