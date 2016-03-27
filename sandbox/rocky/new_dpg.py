from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.sampler.utils import rollout
from lasagne.updates import adam
from rllab.misc.ext import AttrDict
import numpy as np
import theano.tensor as TT
import theano


# Manually construct the policy and Q function

class Policy(object):
    def __init__(self, obs_dim, action_dim, h1_size=32, h2_size=33):
        W1 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(obs_dim, h1_size)) / np.sqrt(obs_dim),
            "W1",
        )
        b1 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, h1_size)) / np.sqrt(obs_dim),
            "b1",
            broadcastable=(True, False),
        )

        W2 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(h1_size, h2_size)) / np.sqrt(h1_size),
            "W2",
        )
        b2 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, h2_size)) / np.sqrt(h1_size),
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
            split_ids = np.insert(np.cumsum(flat_dims)[:-1], 0)
            import ipdb; ipdb.set_trace()


        self.get_param_values = get_param_values
        self.set_param_values = set_param_values

        obs_var = TT.matrix("obs")

        def action_var(obs_var):
            h1 = relu(obs_var.dot(W1) + b1)
            h2 = relu(h1.dot(W2) + b2)
            output = tanh(h2.dot(W3) + b3)
            return output

        self.action_var = action_var
        self.get_action = theano.function(
            inputs=[obs_var],
            outputs=action_var(obs_var),
            allow_input_downcast=True
        )


class QFunction(object):
    def __init__(self, obs_dim, action_dim, h1_size=32, h2_size=33):
        W1 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(obs_dim, h1_size)) / np.sqrt(obs_dim),
            "W1",
        )
        b1 = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, h1_size)) / np.sqrt(obs_dim),
            "b1",
            broadcastable=(True, False),
        )

        W2_obs = theano.shared(
            np.random.uniform(low=-1, high=1, size=(h1_size, h2_size)) / np.sqrt(h1_size),
            "W2_obs",
        )
        b2_obs = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, h2_size)) / np.sqrt(h1_size),
            "b2_obs",
            broadcastable=(True, False),
        )

        W2_action = theano.shared(
            np.random.uniform(low=-1, high=1, size=(action_dim, h2_size)) / np.sqrt(action_dim),
            "W2_action",
        )
        b2_action = theano.shared(
            np.random.uniform(low=-1, high=1, size=(1, h2_size)) / np.sqrt(h1_size),
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
            split_ids = np.insert(np.cumsum(flat_dims)[:-1], 0)
            import ipdb; ipdb.set_trace()


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
        self._observations = np.zeros((max_pool_size, obs_dim))
        self._actions = np.zeros((max_pool_size, action_dim))
        self._next_observations = np.zeros((max_pool_size, obs_dim))
        self._rewards = np.zeros((max_pool_size,))
        self._terminals = np.zeros((max_pool_size,))
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


class OUStrategy(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, dt=1.):
        self._action_dim = action_dim
        self._mu = mu
        self._theta = theta
        self._sigma = sigma
        self._dt = dt
        self._state = None
        self.reset()

    def reset(self):
        self._state = np.zeros(self._action_dim)

    def evolve_state(self):
        dx = self._theta * (self._mu - self._state) * self._dt + \
             self._sigma * np.random.normal(size=(self._action_dim,)) * self._dt
        self._state = self._state + dx
        return self._state

    def get_action(self, observation, policy):
        return policy.get_action(observation) + self.evolve_state()


def test_ou_strategy():
    ou = OUStrategy(action_dim=1, mu=0, theta=0.15, sigma=0.2)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt
    plt.plot(states)
    plt.show()

test_ou_strategy()


def run():
    n_epochs = 200
    n_epoch_itrs = 1000
    min_pool_size = 1000
    max_pool_size = 1000000
    max_path_length = 100
    batch_size = 32
    n_eval_trajs = 10

    env = normalize(CartpoleEnv())
    obs_dim = env.observation_space.flat_dim
    action_dim = env.action_space.flat_dim
    policy = Policy(obs_dim=obs_dim, action_dim=action_dim)
    qf = QFunction(obs_dim=env.observation_space.flat_dim, action_dim=env.action_space.flat_dim)
    target_policy = Policy(obs_dim=obs_dim, action_dim=action_dim)
    target_qf = QFunction(obs_dim=env.observation_space.flat_dim, action_dim=env.action_space.flat_dim)
    es = OUStrategy(action_dim=action_dim)
    pool = ReplayPool(max_pool_size=max_pool_size, obs_dim=obs_dim, action_dim=action_dim)

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
        inputs=[obs_var, action_var, q_var],
        outputs=q_loss,
        updates=adam(q_loss, qf.params),
    )

    terminal = True
    obs = None
    t = 0

    for epoch in xrange(n_epochs):
        for epoch_itr in xrange(n_epoch_itrs):
            if terminal:
                es.reset()
                obs = env.reset()
                t = 0
                terminal = False

            action = es.get_action(obs, policy)
            next_obs, reward, terminal, _ = env.step(action)
            t += 1

            if t >= max_path_length:
                terminal = True

            pool.add_sample(obs, action, next_obs, reward, terminal)

            if pool.size >= min_pool_size:
                # train policy
                batch = pool.random_batch(batch_size)
                target_q = f_target_q(batch["next_observations"])
                ys = batch["rewards"] + (1 - batch["terminals"]) * target_q

        paths = []
        for _ in xrange(n_eval_trajs):
            path = rollout(
                env,
                policy,
                max_length=max_path_length,
            )
            paths.append(path)
        avg_return = np.mean([np.sum(p["rewards"]) for p in paths])
        print "Average Return: %f" % avg_return
