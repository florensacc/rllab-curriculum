import dqn
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.sampler.utils import rollout
from rllab.misc.ext import AttrDict
import numpy as np


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

    def get_action(self, observation, network):
        return network.select_action(observation) + self.evolve_state()


def run():
    n_epochs = 200
    n_epoch_itrs = 1000
    min_pool_size = 10000
    max_pool_size = 1000000
    max_path_length = 100
    n_eval_trajs = 10
    discount = 0.99

    env = normalize(CartpoleEnv())
    obs_dim = env.observation_space.flat_dim
    action_dim = env.action_space.flat_dim
    es = OUStrategy(action_dim=action_dim)

    network = dqn.DQN(
        state_size=obs_dim,
        action_size=action_dim,
        replay_pool_size=max_pool_size,
        memory_threshold=min_pool_size,
        actor_learning_rate=1e-5,
        critic_learning_rate=5e-5,
        discount=discount,
    )

    terminal = True
    obs = None
    t = 0

    for epoch in xrange(n_epochs):
        print "Epoch %d" % epoch
        for epoch_itr in xrange(n_epoch_itrs):
            if terminal:
                es.reset()
                obs = env.reset()
                t = 0
                terminal = False

            action = es.get_action(obs, network)
            next_obs, reward, terminal, _ = env.step(action)
            t += 1

            if t >= max_path_length:
                terminal = True

            if terminal:
                network.add_transition(obs, action, reward, None)
            else:
                network.add_transition(obs, action, reward, next_obs)

            network.update()
        # test performance
        paths = []
        for _ in xrange(n_eval_trajs):
            path = rollout(
                env,
                AttrDict(
                    reset=lambda: None,
                    get_action=lambda obs_: (network.select_action(obs_), dict())
                ),
                max_length=max_path_length,
            )
            paths.append(path)
        avg_return = np.mean([np.sum(p["rewards"]) for p in paths])
        print "Average Return: %f" % avg_return
        print "Actor net"
        network.actor_net.print_diagnostics()
        print "Critic net"
        network.critic_net.print_diagnostics()
        print "Actor target net"
        network.actor_target_net.print_diagnostics()
        print "Critic target net"
        network.critic_target_net.print_diagnostics()


run()
