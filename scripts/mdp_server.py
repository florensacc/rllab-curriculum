import zmq
import argparse
import numpy as np
from rllab.mdp.base import MDP
from rllab.misc.resolve import load_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mdp', type=str, help='module path to the mdp class')
    #parser.add_argument('port', type=str, help='port for the server')
    args = parser.parse_args()
    mdp = load_class(args.mdp, MDP, ["rllab", "mdp"])()

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
                state, obs = mdp.reset()
                message = ",".join(map(str, state.flat)) + ";" + ",".join(map(str, obs.flat))
                socket.send(message)
            elif msg.startswith('step'):
                _, state_str, action_str = msg.split(';')
                state = np.array(map(float, state_str.split(",")))
                action = np.array(map(float, action_str.split(",")))
                next_state, next_action, reward, terminal = mdp.step(state, action)
                socket.send("%s;%s;%s;%s" % (
                    ",".join(map(str, next_state)),
                    ",".join(map(str, next_action)),
                    str(reward),
                    str(int(terminal))
                ))
            elif msg == 'action_dim':
                socket.send(str(mdp.action_dim))
            elif msg == 'observation_dim':
                socket.send(str(mdp.observation_shape[0]))
            elif msg == 'action_bounds':
                lb, ub = mdp.action_bounds
                message = ",".join(map(str, lb)) + ";" + ",".join(map(str, ub))
                socket.send(message)
            else:
                raise 'unknown'
                socket.send('unknown')
        except zmq.Again:
            continue
        except Exception as e:
            socket.send(str(e))
