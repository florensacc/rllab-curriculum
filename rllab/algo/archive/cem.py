import theano.tensor as T
import numpy as np
from misc.console import log
import time
from multiprocessing import Process, Queue, Pool
from Queue import Empty

def cem(f, x0, init_std, n_samples=100, n_iter=100, best_frac=0.05, extra_std=1.0, extra_decay_time=100):

    cur_std = init_std
    cur_mean = x0
    K = cur_mean.size
    n_best = int(n_samples * best_frac)

    for itr in range(n_iter):
        # sample around the current distribution
        extra_var_mult = max(1.0 - itr / extra_decay_time, 0)
        sample_std = np.sqrt(np.square(cur_std) + np.square(extra_std) * extra_var_mult)
        xs = np.random.randn(n_samples, K) * sample_std.reshape(1, -1) + cur_mean.reshape(1, -1)
        fs = np.array(map(f, xs))
        best_inds = (-fs).argsort()[:n_best]
        best_xs = xs[best_inds]
        cur_mean = best_xs.mean(axis=0)
        cur_std = best_xs.std(axis=0)
        best_x = best_xs[0]
        yield cur_mean, cur_std, np.mean(fs), best_x, np.max(fs)

def rollout(policy, param_val, mdp, discount):
    prev_x = policy.get_param_values()
    policy.set_param_values(param_val)
    state, obs = mdp.reset()
    #timestep = 5.0 / 1000
    rewards = []
    for _ in range(mdp.horizon):
        action, action_prob = policy.get_action(obs)
        next_state, next_obs, reward, done = mdp.step(state, action)
        rewards.append(reward)
        if done:
            state, obs = mdp.reset()#next_state, next_obs
            break
        state, obs = next_state, next_obs
        mdp.plot()
    ret = 0
    for reward in rewards[::-1]:
        ret = ret*discount + reward

    print 'demo reward: %f' % ret
        #time.sleep(timestep)#mdp.timestep)
    policy.set_param_values(prev_x)

#def rollout

def mk_eval_policy(policy, mdp, max_steps_per_traj, discount):
    def f(x):
        prev_x = policy.get_param_values()
        policy.set_param_values(x)
        state, obs = mdp.reset()
        ret = 0
        rewards = []
        #mdp.start_viewer()
        for _ in range(max_steps_per_traj):
            action, action_prob = policy.get_action(obs)
            next_state, next_obs, reward, done = mdp.step(state, action)
            #mdp.viewer.loop_once()
            rewards.append(reward)
            if done:
                break
            state, obs = next_state, next_obs
        ret = 0
        for reward in rewards[::-1]:
            ret = ret*discount + reward
        policy.set_param_values(prev_x)
        #mdp.stop_viewer()
        return ret
    return f



class CEM(object):

    def __init__(
            self,
            max_steps_per_traj=100,
            samples_per_itr=100,
            n_itr=100,
            best_frac=0.1,
            extra_std=1.0,
            extra_decay_time=100,
            exp_name='cem',
            discount=0.99):
        self.max_steps_per_traj = max_steps_per_traj
        self.samples_per_itr = samples_per_itr
        self.n_itr = n_itr
        self.best_frac = best_frac
        self.extra_std = extra_std
        self.extra_decay_time = extra_decay_time
        self.exp_name = exp_name
        self.discount = discount

    def train(self, mdp, policy):

        x0 = policy.get_param_values()
        init_std = np.ones(x0.shape) * 10

        f = mk_eval_policy(policy, mdp, self.max_steps_per_traj, self.discount)

        can_demo = getattr(mdp, 'start_viewer', None) is not None

        def start_mdp_viewer(_mdp, _policy, queue):
            mdp = _mdp
            policy = _policy
            mdp.start_viewer()
            try:
                while True:
                    msg = None
                    while True:
                        try:
                            msg = queue.get_nowait()
                        except Empty:
                            break
                    if msg:
                        if msg[0] == 'stop':
                            break
                        elif msg[0] == 'demo':
                            itr, cur_mean = msg[1:]
                            print('demoing itr %d' % itr)
                            rollout(policy, cur_mean, mdp, self.discount)
                        elif msg[0] == 'loop':
                            itr, cur_mean = msg[1:]
                            print('demoing itr %d' % itr)
                            while True:
                                rollout(policy, cur_mean, mdp, self.discount)

            except KeyboardInterrupt:
                pass
            mdp.stop_viewer()

        if can_demo:
            q = Queue()
            p = Process(target=start_mdp_viewer, args=(mdp, policy, q))
            p.start()

        for itr, data in enumerate(cem(f, x0, init_std)):
            cur_mean, cur_std, avg_reward, best_x, max_reward = data
            log('itr %d: avg performance %f; best performance %f; median std: %f' % (itr, avg_reward, max_reward, np.median(cur_std)))
            if can_demo:
                q.put(['demo', itr, best_x])

        if can_demo:
            q.put(['loop', itr, policy, best_x])
