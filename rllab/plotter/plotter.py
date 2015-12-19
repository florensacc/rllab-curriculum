import atexit
from Queue import Empty
from multiprocessing import Process, Queue
from rllab.sampler.utils import rollout
import numpy as np

__all__ = [
    'init_worker',
    'init_plot',
    'update_plot'
]

process = None
queue = None


def _worker_start():
    mdp = None
    policy = None
    try:
        while True:
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                break
            if 'update' in msgs:
                mdp, policy = msgs['update']
                mdp.start_viewer()
            if 'demo' in msgs:
                param_values, max_length = msgs['demo']
                policy.set_param_values(param_values)
                rollout(mdp, policy, max_length=max_length, animated=True)
    except KeyboardInterrupt:
        pass
    if mdp:
        mdp.stop_viewer()


def _shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.close()
        process.join()


def init_worker():
    global process, queue
    queue = Queue()
    process = Process(target=_worker_start)
    process.start()
    atexit.register(_shutdown_worker)


def init_plot(mdp, policy):
    queue.put(['update', mdp, policy])


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
