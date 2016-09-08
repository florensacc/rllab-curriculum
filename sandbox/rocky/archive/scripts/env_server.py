import zmq
import argparse
import numpy as np
import msgpack
from rllab.envs.base import Env
from rllab.misc.resolve import load_class


class SerializeEnv(object):

    def __init__(self, env):
        self._env = env

    def reset(self):
        obs = self._env.reset()
        return msgpack.packb(self._env.observation_space.flatten(obs).tolist())

    def step(self, action):
        next_obs, reward, done, env_info = self._env.step(action)
        next_obs_list = self._env.observation_space.flatten(next_obs).tolist()
        env_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, help='module path to the env class')
    parser.add_argument('--port', type=int, help='initial port to bind to')
    args = parser.parse_args()
    env = load_class(args.env, Env, ["rllab", "env"])()

    context = zmq.Context()
    init_socket = context.socket(zmq.REP)
    init_socket.bind('tcp://*:4265')
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port('tcp://*')
    init_socket.recv()
    init_socket.send(str(port))
    init_socket.close()
    socket.linger = 1000000
    while True:
        try:
            msg = socket.recv()
            if msg == 'reset':
                state, obs = env.reset()
                message = ",".join(map(str, state.flat)) + ";" + ",".join(map(str, obs.flat))
                socket.send(message)
            elif msg.startswith('step'):
                _, state_str, action_str = msg.split(';')
                state = np.array(list(map(float, state_str.split(","))))
                action = np.array(list(map(float, action_str.split(","))))
                next_state, next_action, reward, terminal = env.step(state, action)
                socket.send("%s;%s;%s;%s" % (
                    ",".join(map(str, next_state)),
                    ",".join(map(str, next_action)),
                    str(reward),
                    str(int(terminal))
                ))
            elif msg == 'action_space':
                socket.send(str(env.action_space.flat_dim))
            elif msg == 'observation_space':
                socket.send(str(env.observation_space.flat_dim))
            elif msg == 'action_bounds':
                lb, ub = env.action_bounds
                message = ",".join(map(str, lb)) + ";" + ",".join(map(str, ub))
                socket.send(message)
            elif msg == 'terminate':
                socket.send(str(env.observation_space.flat_dim))
            else:
                raise 'unknown'

        except zmq.Again:
            continue
        except Exception as e:
            socket.send(str(e))
