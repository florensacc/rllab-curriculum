from sandbox.rocky.hogwild.shared_policy import SharedPolicy
from sandbox.rocky.hogwild.shared_q_function import
from sandbox.rocky.hogwild.async_ddpg import AsyncDDPG
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from contextlib import closing
import multiprocessing as mp
import numpy as np

shared_policy = None


def worker_run(shared_T, shared_policy_, child_conn):
    global shared_policy
    shared_policy = shared_policy_
    child_conn.recv()
    param = shared_policy.get_params()[0].get_value(borrow=True)
    child_conn.send(np.copy(param))


def get_first_param():
    pass


def main():
    env = CartpoleEnv()
    policy = DeterministicMLPPolicy(env.spec)
    shared_policy = SharedPolicy(policy)
    qf = ContinuousMLPQFunction(env.spec)
    shared_qf =

    n_child_processes = 4

    pipes = [mp.Pipe() for _ in xrange(n_child_processes)]
    processes = [mp.Process(target=worker_run, args=(shared_policy, child_conn)) for _, child_conn in pipes]

    for p in processes:
        p.start()

    print "All processes started"

    param = shared_policy.get_params()[0].get_value(borrow=True)
    np.copyto(param, np.ones_like(param))

    for parent_conn, _ in pipes:
        parent_conn.send(None)

    print "Message sent"

    for parent_conn, _ in pipes:
        print parent_conn.recv()

    print "Message received"


        # with closing(mp.Pool(initializer=init_worker, initargs=(shared_policy,))) as p:
        #     print p.map(get_first_param, xrange(10))
        #     pass
        # p.map(update_x, xrange(10000))


if __name__ == "__main__":
    main()
