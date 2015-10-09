import theano.tensor as T
import numpy as np
from misc.console import log

def cem(f, x0, init_std, n_samples=100, n_iter=100, best_frac=0.1, extra_std=1.0, extra_decay_time=50):

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
        yield cur_mean, cur_std, max(fs)

def mk_eval_policy(policy, mdp, max_steps_per_traj):
    def f(x):
        prev_x = policy.get_param_values()
        policy.set_param_values(x)
        state, obs = mdp.sample_initial_state()
        ret = 0
        n_steps = 0
        while n_steps < max_steps_per_traj:
            action, action_prob = policy.get_actions_single(obs)
            next_state, next_obs, reward, done, steps = mdp.step_single(state, action)
            n_steps += steps
            ret += reward
            if done:
                break
            state, obs = next_state, next_obs
        policy.set_param_values(prev_x)
        return ret
    return f



class CEM(object):

    def __init__(
            self,
            max_steps_per_traj=1000,
            samples_per_itr=100,
            n_itr=100,
            best_frac=0.1,
            extra_std=1.0,
            extra_decay_time=50):
        self.max_steps_per_traj = max_steps_per_traj
        self.samples_per_itr = samples_per_itr
        self.n_itr = n_itr
        self.best_frac = best_frac
        self.extra_std = extra_std
        self.extra_decay_time = extra_decay_time

    def train(self, gen_mdp, gen_policy):
        mdp = gen_mdp()
        input_var = T.matrix('input')  # N*Ds
        policy = gen_policy(input_var, mdp)

        x0 = policy.get_param_values()
        init_std = np.ones(x0.shape)

        f = mk_eval_policy(policy, mdp, self.max_steps_per_traj)
        for itr, data in enumerate(cem(f, x0, init_std)):
            cur_mean, cur_std, best_reward = data
            log('itr %d: best performance %f' % (itr, best_reward))
