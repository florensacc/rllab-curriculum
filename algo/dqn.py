from __future__ import print_function
from pylearn2.training_algorithms.learning_rule import RMSProp
import numpy as np
import tensorfuse as theano 
import tensorfuse.tensor as T
import itertools
import time
import sys
#from misc.console import log

def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.cast['float32'](0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

class DQN(object):

    def __init__(
            self,
            replay_pool_size=1000000,
            min_pool_size=50000,
            exploration_decay_range=100000,
            learning_rate=1e-4,
            max_epsilon=1,
            min_epsilon=0.1,
            batch_size=32,
            discount=0.99):
        self.replay_pool_size = replay_pool_size
        self.min_pool_size = min_pool_size
        self.exploration_decay_range = exploration_decay_range
        self.learning_rate = learning_rate
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.discount = discount
        pass

    def train(self, gen_mdp, gen_q_func):
        # the pool is organized as a circular buffer, where pool_ptr
        # denote the next position to update in the pool, and it's
        # updated according to
        # pool_ptr = (pool_ptr + 1) % replay_pool_size
        # this ensures that only the most recent experiences are kept,
        # while still allowing us to use continuous memory storage (numpy arrays)
        pool_size = 0
        pool_ptr = 0

        mdp = gen_mdp()
        #states, obs = mdp.sample_initial_state()
        Ds = mdp.observation_shape[0]
        Da = mdp.n_actions
        # initialize memory
        # A single experience is a tuple (s, s', a, r, done), concatenated to be a
        # single row vector of dimension Ds+Ds+1+1
        replay_pool = np.zeros((self.replay_pool_size, Ds+Ds+1+1+1))
        # initialize Q function

        input_var = T.matrix("input") # N*Ds

        action_var = T.vector("action", dtype='uint8')
        fit_var = T.vector("Y")

        log = print

        log("Compiling q functions...")

        Q = gen_q_func(mdp.observation_shape, mdp.n_actions, input_var)
        Q_tgt = gen_q_func(mdp.observation_shape, mdp.n_actions, input_var)

        N_var = input_var.shape[0]

        #Q_func = theano.function([X], Q.)
        #Q = new_q_function(X, mdp)

        loss = T.mean(T.square(Q.q_var[T.arange(N_var), action_var] - fit_var))# / batch_size

        log("compiling loss function...")
        loss_function = theano.function([input_var, fit_var, action_var], loss)

        grads = T.grad(loss, Q.params)

        #rmsprop = RMSProp(decay=0.95)
        #updates = rmsprop.get_updates(
        #        learning_rate=self.learning_rate,
        #        grads=dict(zip(Q.params, grads)))

        updates = Adam(loss, Q.params, lr=self.learning_rate)

        log("compiling update function...")
        do_update = theano.function([input_var, fit_var, action_var], [], updates=updates)

        state, obs = mdp.reset()

        n_episodes = 0
        cur_total_reward = 0
        avg_total_reward = 0
        moving_avg_total_reward = 0
        moving_factor = 0.05
        n_episodes_sliding = 0
        avg_total_reward_sliding = 0

        for itr in itertools.count():
            epsilon = np.clip(
                np.interp(itr, [self.min_pool_size, self.exploration_decay_range], [self.max_epsilon, self.min_epsilon]),
                self.min_epsilon,
                self.max_epsilon
            )

            # choose action
            a_idx = None
            if np.random.rand() < epsilon:
                a_idx = np.random.choice(range(Da))
            else:
                q_val = Q_tgt.compute_q_val([obs])[0]
                a_idx = np.argmax(q_val)
            
            #r = 0
            
            next_state, next_obs, reward, done, step = mdp.step_single(state, [a_idx])

            cur_total_reward += reward
            
            if done:
                n_episodes = n_episodes + 1
                n_episodes_sliding = n_episodes_sliding + 1
                avg_total_reward += 1.0 / n_episodes * (cur_total_reward - avg_total_reward)
                moving_avg_total_reward += moving_factor * (cur_total_reward - moving_avg_total_reward)
                avg_total_reward_sliding += 1.0 / n_episodes_sliding * (cur_total_reward - avg_total_reward_sliding)
                cur_total_reward = 0

                
            if itr % 1000 == 0:
                log('iteration\t#%d; epsilon:\t%.3f; #episodes:\t%d; avg total reward:\t%.3f; moving avg total reward:\t%.3f; avg total reward for last 1000 iterations:\t%.3f' % (itr, epsilon, n_episodes, avg_total_reward, moving_avg_total_reward, avg_total_reward_sliding))
                #print 'epsilon: %f' % epsilon
                #print '#episodes: %d' % n_episodes
                #print 'avg total reward: ', avg_total_reward
                #print 'avg total reward for last 1000 iterations: ', avg_total_reward_sliding
                avg_total_reward_sliding = 0
                n_episodes_sliding = 0
                
                
            #r = min(1, max(-1, r))
            
            experience = np.concatenate([
                obs,
                next_obs,
                [a_idx],
                [reward],
                [int(done)],
            ])
            replay_pool[pool_ptr, :] = experience
            pool_ptr = (pool_ptr + 1) % self.replay_pool_size
            pool_size = min(pool_size + 1, self.replay_pool_size)

            state, obs = next_state, next_obs
            
            if pool_size >= self.min_pool_size:# and itr % 100000 == 0:
                batch = replay_pool[np.random.randint(pool_size, size=self.batch_size)]
                s_batch = batch[:, :Ds]
                next_s_batch = batch[:, Ds:Ds+Ds]
                a_batch = np.uint8(batch[:, Ds+Ds])
                # clip reward
                r_batch = np.clip(batch[:, Ds+Ds+1], -1, 1)
                dones_batch = batch[:, Ds+Ds+2]
                y_batch = r_batch + self.discount * np.max(Q_tgt.compute_q_val(next_s_batch), axis=1) * (1 - dones_batch)
                #for _ in range(100):
                do_update(s_batch, y_batch, a_batch)
                if itr % 10000 == 0:
                    Q_tgt.set_param_values(Q.get_param_values())
                    log('loss: %f' % loss_function(s_batch, y_batch, a_batch))
                    
                    # play the game according to the policy once
                    _t = time.time()
                    avg_rs = []
                    max_q_val = -np.inf
                    for _ in range(100):
                        policy_state, policy_obs = mdp.reset()
                        policy_r = 0
                        t = 0
                        while True:
                            #if np.random.rand() < 0.05:
                            #    a_idx = np.random.choice(range(Da))
                            #else:
                            q_val = Q_tgt.compute_q_val([policy_obs])[0]
                            max_q_val = max(np.max(q_val), max_q_val)
                            a_idx = np.argmax(q_val)
                            policy_state, policy_obs, reward, done, step = mdp.step_single(policy_state, [a_idx])
                            policy_r += reward
                            t += 1
                            if done:
                                break
                            if t > 100: # 5 min max
                                break
                        avg_rs.append(policy_r)
                    log('on-policy reward for 100 trials: %.3f; executing policy took %.2fs' % (np.mean(avg_rs), time.time() - _t))
                    print('max q val: %f', max_q_val)
                    sys.stdout.flush()

            #if itr % saveout_period == 0 and itr > 0:
            #    print 'saving out parameters....'
            #    import utils
            #    utils.save_out_results(('%s_%s_itr_%d.h5') % (rom_name, exp_name, itr), itr, replay_pool, pool_ptr, pool_size, Q)
