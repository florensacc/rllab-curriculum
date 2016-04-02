from __future__ import print_function

import cPickle as pickle
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as TT

from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc import logger
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy


# from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv


def extract(x, *keys):
    assert isinstance(x, list)
    return tuple([xi[k] for xi in x] for k in keys)


def merge_dict(*args):
    if any([isinstance(x, OrderedDict) for x in args]):
        z = OrderedDict()
    else:
        z = dict()
    for x in args:
        z.update(x)
    return z


# class logger(object):
#     tabulars = list()
#
#     @classmethod
#     def record_tabular(cls, key, val):
#         cls.tabulars.append((key, val))
#
#     @classmethod
#     def dump_tabular(cls):
#         print("=============================")
#         for k, v in cls.tabulars:
#             print(k, "\t", v)
#         cls.tabulars = list()


def rollout(env, agent, max_length=np.inf):
    observations = []
    actions = []
    rewards = []
    o = env.reset()
    agent.reset()
    path_length = 0
    while path_length < max_length:
        a, _ = agent.get_action(o)
        next_o, r, d, _ = env.step(a)
        observations.append(o)
        rewards.append(r)
        actions.append(a)
        path_length += 1
        if d:
            break
        o = next_o
    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
    )


def adam(loss, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = theano.grad(loss, params)
    t_prev = theano.shared(np.float32(0.))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate * TT.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * g_t ** 2
        step = a_t * m_t / (TT.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates


class CartpoleEnv(object):
    def __init__(self):
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.max_force = 10.
        self.dt = .05
        self._state = None
        self.reset()

    def reset(self):
        bounds = np.array([
            self.max_cart_speed,
            self.max_cart_speed,
            self.max_pole_speed,
            self.max_pole_speed
        ])

        low, high = -0.05 * bounds, 0.05 * bounds
        self._state = np.random.uniform(low, high)
        return self._state

    @property
    def observation_dim(self):
        return 4

    @property
    def action_dim(self):
        return 1

    def step(self, action):
        state = self._state
        u0 = action[0]
        u0 = np.clip(u0 * self.max_force, -self.max_force, self.max_force)

        dt = self.dt

        z, zdot, th, thdot = state

        th1 = np.pi - th

        g = 10.
        mc = 1.  # mass of cart
        mp = .1  # mass of pole
        muc = .0005  # coeff friction of cart
        mup = .000002  # coeff friction of pole
        l = 1.  # length of pole

        thddot = -(-g * np.sin(th1)
                   + np.cos(th1) * (-u0 - mp * l * thdot ** 2 * np.sin(th1) + muc * np.sign(zdot)) / (mc + mp)
                   - mup * thdot / (mp * l)) \
                 / (l * (4 / 3. - mp * np.cos(th1) ** 2 / (mc + mp)))
        zddot = (u0 + mp * l * (thdot ** 2 * np.sin(th1) - thddot * np.cos(th1)) - muc * np.sign(zdot)) \
                / (mc + mp)

        newzdot = zdot + dt * zddot
        newz = z + dt * newzdot
        newthdot = thdot + dt * thddot
        newth = th + dt * newthdot

        done = (newz > self.max_cart_pos) | (newz < -self.max_cart_pos) | (newth > self.max_pole_angle) | (
            newth < -self.max_pole_angle)

        ucost = 1e-5 * (u0 ** 2)
        xcost = 1 - np.cos(th)
        notdone = 1 - done

        reward = notdone * 10 - notdone * xcost - notdone * ucost
        self._state = np.array([newz, newzdot, newth, newthdot])
        return self._state, reward, done, dict()


# Manually construct the policy and Q function

class Policy(object):
    def __init__(self, obs_dim, action_dim, h1_size=400, h2_size=300):
        W1 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(obs_dim, h1_size)) / np.sqrt(obs_dim),# * np.sqrt(3),
            "W1",
        )
        b1 = theano.shared(
            np.zeros((1, h1_size)),
            "b1",
            broadcastable=(True, False),
        )

        W2 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(h1_size, h2_size)) / np.sqrt(h1_size),# * np.sqrt(3),
            "W2",
        )
        b2 = theano.shared(
            np.zeros((1, h2_size)),
            "b2",
            broadcastable=(True, False),
        )

        W3 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(h2_size, action_dim)) * 3e-3,
            "W3",
        )
        b3 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, action_dim)) * 3e-3,
            "b3",
            broadcastable=(True, False),
        )

        relu = TT.nnet.relu
        tanh = TT.tanh

        self.params = params = [W1, b1, W2, b2, W3, b3]
        self.param_shapes = param_shapes = [x.get_value().shape for x in params]
        self.param_dtypes = [x.get_value().dtype for x in params]

        def get_param_values():
            return np.concatenate([x.get_value().flatten() for x in self.params])

        def set_param_values(flat_val):
            flat_dims = map(np.prod, param_shapes)
            split_ids = np.cumsum(flat_dims)[:-1]
            new_vals = np.split(flat_val, split_ids)
            for p, val, shape in zip(self.params, new_vals, param_shapes):
                p.set_value(val.reshape(shape))

        self.get_param_values = get_param_values
        self.set_param_values = set_param_values

        obs_var = TT.matrix("obs")

        def action_var(obs_var):
            h1 = relu(obs_var.dot(W1) + b1)
            h2 = relu(h1.dot(W2) + b2)
            output = tanh(h2.dot(W3) + b3)
            return output

        self.action_var = action_var
        f_action = theano.function(
            inputs=[obs_var],
            outputs=action_var(obs_var),
            allow_input_downcast=True
        )

        def get_action(obs):
            return f_action([obs])[0], dict()

        self.get_action = get_action

    def reset(self):
        pass


class QFunction(object):
    def __init__(self, obs_dim, action_dim, h1_size=400, h2_size=300):
        W1 = theano.shared(
            np.random.normal(low=-1, high=1, size=(obs_dim, h1_size)) / np.sqrt(obs_dim),# * np.sqrt(3),
            "W1",
        )
        b1 = theano.shared(
            np.zeros((1, h1_size)),
            "b1",
            broadcastable=(True, False),
        )

        W2_obs = theano.shared(
            np.random.uniform(low=-1, high=1, size=(h1_size, h2_size)) / np.sqrt(h1_size) * np.sqrt(3),
            "W2_obs",
        )
        b2_obs = theano.shared(
            np.zeros((1, h2_size)),
            "b2_obs",
            broadcastable=(True, False),
        )

        W2_action = theano.shared(
            np.random.uniform(low=-1, high=1, size=(action_dim, h2_size)) / np.sqrt(action_dim),# * np.sqrt(3),
            "W2_action",
        )
        b2_action = theano.shared(
            np.zeros((1, h2_size)),
            "b2_action",
            broadcastable=(True, False),
        )

        W3 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(h2_size, 1)) * 3e-3,
            "W3",
        )
        b3 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, 1)) * 3e-3,
            "b3",
            broadcastable=(True, False),
        )

        relu = TT.nnet.relu

        self.params = params = [W1, b1, W2_obs, b2_obs, W2_action, b2_action, W3, b3]

        self.param_shapes = param_shapes = [x.get_value().shape for x in params]
        self.param_dtypes = [x.get_value().dtype for x in params]

        def get_param_values():
            return np.concatenate([x.get_value().flatten() for x in self.params])

        def set_param_values(flat_val):
            flat_dims = map(np.prod, param_shapes)
            split_ids = np.cumsum(flat_dims)[:-1]
            new_vals = np.split(flat_val, split_ids)
            for p, val, shape in zip(self.params, new_vals, param_shapes):
                p.set_value(val.reshape(shape))

        self.get_param_values = get_param_values
        self.set_param_values = set_param_values

        obs_var = TT.matrix("obs")
        action_var = TT.matrix("action")

        def q_var(obs_var, action_var):
            h1 = relu(obs_var.dot(W1) + b1)
            a2_obs = h1.dot(W2_obs) + b2_obs
            a2_action = action_var.dot(W2_action) + b2_action
            h2 = relu(a2_obs + a2_action)
            output = h2.dot(W3) + b3
            return output.flatten()

        self.q_var = q_var
        self.get_q = theano.function(
            inputs=[obs_var, action_var],
            outputs=q_var(obs_var, action_var),
            allow_input_downcast=True
        )


class ReplayPool(object):
    def __init__(self, max_pool_size, obs_dim, action_dim):
        self._observations = np.ones((max_pool_size, obs_dim)) * np.nan
        self._actions = np.ones((max_pool_size, action_dim)) * np.nan
        self._next_observations = np.ones((max_pool_size, obs_dim)) * np.nan
        self._rewards = np.ones((max_pool_size,)) * np.nan
        self._terminals = np.ones((max_pool_size,)) * np.nan
        self._size = 0
        self._top = 0
        self._max_pool_size = max_pool_size

    def add_sample(self, obs, action, next_obs, reward, terminal):
        top = self._top
        self._observations[top] = obs
        self._actions[top] = action
        self._next_observations[top] = next_obs
        self._rewards[top] = reward
        self._terminals[top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        self._size = min(self._max_pool_size, self._size + 1)

    def random_batch(self, batch_size):
        assert self._size >= batch_size
        ids = np.random.randint(low=0, high=self._size, size=batch_size)
        return dict(
            observations=self._observations[ids],
            actions=self._actions[ids],
            next_observations=self._next_observations[ids],
            rewards=self._rewards[ids],
            terminals=self._terminals[ids],
        )

    @property
    def size(self):
        return self._size


# class OUStrategy(object):
#     def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, dt=1.):
#         self._action_dim = action_dim
#         self._mu = mu
#         self._theta = theta
#         self._sigma = sigma
#         self._dt = dt
#         self._state = None
#         self.reset()
#
#     def reset(self):
#         self._state = np.zeros(self._action_dim)
#
#     def evolve_state(self):
#         dx = self._theta * (self._mu - self._state) * self._dt + \
#              self._sigma * np.random.normal(size=(self._action_dim,)) * self._dt
#         self._state += dx
#         return self._state
#
#     def get_action(self, observation, policy):
#         # return np.random.uniform(-1, 1, (self._action_dim,))
#         return np.clip(policy.get_action(observation)[0] + self.evolve_state(), -1., 1.), dict()
#
#
# class EpsilonGreedyStrategy(object):
#     def __init__(self, action_dim, epsilon=0.1):
#         self._action_dim = action_dim
#         self._epsilon = epsilon
#
#     def reset(self):
#         pass
#
#     def get_action(self, observation, policy):
#         if np.random.uniform() < self._epsilon:
#             return np.random.uniform(low=-1, high=1, size=self._action_dim), dict()
#         else:
#             return policy.get_action(observation)
#
#
# class GaussianStrategy(object):
#     def __init__(self, action_dim, epsilon=0.1):
#         self._action_dim = action_dim
#         self._epsilon = epsilon
#
#     def reset(self):
#         pass
#
#     def get_action(self, observation, policy):
#         if np.random.uniform() < self._epsilon:
#             return np.random.uniform(low=-1, high=1, size=self._action_dim), dict()
#         else:
#             return policy.get_action(observation)
#
#
# def test_ou_strategy():
#     ou = OUStrategy(action_dim=1, mu=0, theta=0.15, sigma=0.2)
#     states = []
#     for i in range(1000):
#         states.append(ou.evolve_state()[0])
#     import matplotlib.pyplot as plt
#     plt.plot(states)
#     plt.show()


class DPGExperiment(object):
    def __init__(
            self,
            n_epochs=200,
            n_epoch_itrs=1000,
            min_pool_size=10000,
            max_pool_size=1000000,
            max_path_length=100,
            batch_size=64,
            n_eval_trajs=10,
            reward_scaling=1.0,
            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            qf_weight_decay=0.01,
            qf_soft_target_tau=1e-3,
            policy_soft_target_tau=1e-3,
            discount=0.99,
            policy_hidden_sizes=(32, 32),
            qf_hidden_sizes=(32, 32)
    ):

        self.n_epochs = n_epochs
        self.n_epoch_itrs = n_epoch_itrs
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.n_eval_trajs = n_eval_trajs
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.reward_scaling = reward_scaling
        self.qf_weight_decay = qf_weight_decay
        self.qf_soft_target_tau = qf_soft_target_tau
        self.policy_soft_target_tau = policy_soft_target_tau
        self.discount = discount
        assert len(policy_hidden_sizes) == 2
        assert len(qf_hidden_sizes) == 2
        self.policy_hidden_sizes = policy_hidden_sizes
        self.qf_hidden_sizes = qf_hidden_sizes

    def run(self, env):
        obs_dim = env.observation_space.flat_dim
        action_dim = env.action_space.flat_dim

        policy = DeterministicMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=self.policy_hidden_sizes,
        )

        policy = Policy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            h1_size=self.policy_hidden_sizes[0],
            h2_size=self.policy_hidden_sizes[1]
        )
        qf = QFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            h1_size=self.qf_hidden_sizes[0],
            h2_size=self.qf_hidden_sizes[1]
        )
        target_policy = Policy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            h1_size=self.policy_hidden_sizes[0],
            h2_size=self.policy_hidden_sizes[1]
        )
        target_qf = QFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            h1_size=self.qf_hidden_sizes[0],
            h2_size=self.qf_hidden_sizes[1]
        )
        # es = OUStrategy(
        #     action_dim=action_dim,
        #     mu=0,
        #     theta=0.15,
        #     sigma=0.3
        # )  # , dt=0.05)
        es = OUStrategy(env_spec=env.spec, theta=0.15, sigma=0.3)
        pool = ReplayPool(
            max_pool_size=self.max_pool_size,
            obs_dim=obs_dim,
            action_dim=action_dim
        )

        # set to the same parameters initially
        target_policy.set_param_values(policy.get_param_values())
        target_qf.set_param_values(qf.get_param_values())

        obs_var = TT.matrix("obs")
        action_var = TT.matrix("action")
        next_obs_var = TT.matrix("next_obs")
        ys_var = TT.vector("ys")

        target_q_var = target_qf.q_var(next_obs_var, target_policy.action_var(next_obs_var))
        f_target_q = theano.function(
            inputs=[next_obs_var],
            outputs=target_q_var,
            allow_input_downcast=True,
        )
        q_var = qf.q_var(obs_var, action_var)

        q_loss = TT.mean(TT.square(q_var - ys_var))

        f_update_q = theano.function(
            inputs=[obs_var, action_var, ys_var],
            outputs=q_loss,
            updates=adam(q_loss, qf.params, learning_rate=self.qf_learning_rate),
            allow_input_downcast=True
        )

        policy_loss = - TT.mean(qf.q_var(obs_var, policy.action_var(obs_var)))

        f_update_policy = theano.function(
            inputs=[obs_var],
            outputs=policy_loss,
            updates=adam(policy_loss, policy.params, learning_rate=self.policy_learning_rate),
            allow_input_downcast=True
        )

        terminal = True
        obs = None
        t = 0

        for epoch in xrange(self.n_epochs):
            target_qs = []
            q_losses = []
            policy_losses = []
            for epoch_itr in xrange(self.n_epoch_itrs):
                if terminal:# or t > self.max_path_length:
                    es.reset()
                    obs = env.reset()
                    t = 0

                action = es.get_action(0, obs, policy)
                next_obs, reward, terminal, _ = env.step(action)
                t += 1

                if t >= self.max_path_length:
                    terminal = True

                pool.add_sample(obs, action, next_obs, reward * self.reward_scaling, terminal)

                obs = next_obs

                if pool.size >= self.min_pool_size:
                    # train policy
                    batch = pool.random_batch(self.batch_size)
                    target_q = f_target_q(batch["next_observations"])
                    ys = batch["rewards"] + self.discount * (1 - batch["terminals"]) * target_q
                    q_losses.append(f_update_q(batch["observations"], batch["actions"], ys))
                    qf.set_param_values(qf.get_param_values() * (1 - self.qf_weight_decay * self.qf_learning_rate))
                    policy_losses.append(f_update_policy(batch["observations"]))
                    target_qs.extend(target_q)
                    target_qf.set_param_values(
                        target_qf.get_param_values() * (1 - self.qf_soft_target_tau) +
                        qf.get_param_values() * self.qf_soft_target_tau
                    )
                    target_policy.set_param_values(
                        target_policy.get_param_values() * (1 - self.policy_soft_target_tau) +
                        policy.get_param_values() * self.policy_soft_target_tau
                    )

            paths = []
            eval_env = pickle.loads(pickle.dumps(env))
            # eval_env._normalize_reward = False
            for _ in xrange(self.n_eval_trajs):
                path = rollout(
                    eval_env,
                    policy,
                    max_length=self.max_path_length,
                )
                paths.append(path)
            avg_return = np.mean([np.sum(p["rewards"]) for p in paths])
            max_return = np.max([np.sum(p["rewards"]) for p in paths])
            min_return = np.min([np.sum(p["rewards"]) for p in paths])
            avg_len = np.mean([len(p["rewards"]) for p in paths])

            logger.record_tabular("Iteration", epoch)
            logger.record_tabular("AverageReturn", avg_return)
            logger.record_tabular("MaxReturn", max_return)
            logger.record_tabular("MinReturn", min_return)
            logger.record_tabular("AveragePathLength", avg_len)
            # if len(target_qs) > 0:
            logger.record_tabular("AverageTargetQ", np.mean(target_qs))
            logger.record_tabular("AverageQLoss", np.mean(q_losses))
            logger.record_tabular("AveragePolicyLoss", np.mean(policy_losses))
            logger.record_tabular("AverageAbsAction",
                                  np.mean(np.concatenate([np.abs(path["actions"]) for path in paths])))
            env.log_diagnostics(paths)
            logger.dump_tabular()


if __name__ == "__main__":
    DPGExperiment(
        reward_scaling=0.01
    ).run()
