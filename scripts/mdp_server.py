import zmq
import argparse
from rllab.mdp.base import MDP
from rllab.misc.resolve import load_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mdp', type=str, help='module path to the mdp class')
    args = parser.parse_args()
    mdp = load_class(args.mdp, MDP, ["rllab", "mdp"])()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # find an empty port
    port = socket.bind_to_random_port('tcp://*')
    print port
    import sys
    sys.stdout.flush()
    while True:
        msg = socket.recv()
        sys.stdout.flush()
        if msg == 'reset':
            state, obs = mdp.reset()
            message = ",".join(map(str, state.flat)) + ";" + ",".join(map(str, obs.flat))
            socket.send(message)
        elif msg == 'action_dim':
            socket.send(str(mdp.action_dim))
        elif msg == 'observation_shape':
            socket.send(",".join(map(str, mdp.observation_shape)))
        elif msg == 'action_bounds':
            lb, ub = mdp.action_bounds
            message = ",".join(map(str, lb)) + ";" + ",".join(map(str, ub))
            socket.send(message)
