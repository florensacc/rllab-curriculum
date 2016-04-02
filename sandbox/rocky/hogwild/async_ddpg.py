from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from sandbox.rocky.hogwild.shared_policy import SharedPolicy
from sandbox.rocky.hogwild.shared_q_function import SharedQFunction
import multiprocessing as mp


def start_worker(algo, worker_id, pipe):
    algo.worker_train(worker_id=worker_id, pipe=pipe)


class AsyncDDPG(RLAlgorithm, Serializable):
    """
    Asynchronous Deep Deterministic Policy Gradient.
    """

    def __init__(self, env, policy, qf, es, n_workers=1, target_policy=None, target_qf=None):
        if not isinstance(policy, SharedPolicy):
            policy = SharedPolicy(policy)
        if not isinstance(qf, SharedQFunction):
            qf = SharedQFunction(qf)
        if target_policy is None:
            target_policy = policy.new_mem_copy()

        Serializable.quick_init(self, locals())
        # assert isinstance(policy, SharedPolicy)
        # assert isinstance(qf, SharedQFunction)
        self.env = env
        self.policy = policy
        self.qf = qf
        self.es = es
        self.n_workers = n_workers

    def train(self):
        processes = []
        for id in xrange(self.n_workers):
            pipe = mp.Pipe()
            p = mp.Process(target=start_worker, args=(self, id, pipe))
            p.start()
            processes.append(p)

    def worker_train(self, worker_id, pipe):
        # Each worker needs to compile their own functions
        self.init_opt()
        pass
