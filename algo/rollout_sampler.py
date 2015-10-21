from joblib.pool import MemmapingPool
from joblib.parallel import SafeFunction
import numpy as np
import os
import theano.tensor as T
import pyprind
from collections import defaultdict
from misc.console import log
from multiprocessing import Manager
from sampler import launch_sampler
import pickle
import scipy.signal

def _init_subprocess(*args):
    if len(args) == 1:
        _, gen_mdp, gen_policy = pickle.loads(args[0])
    else:
        gen_mdp, gen_policy = args
    global mdp
    global policy
    np.random.seed(os.getpid())
    mdp, policy = _init_mdp_policy(gen_mdp, gen_policy)


def _init_mdp_policy(gen_mdp, gen_policy):
    mdp = gen_mdp()
    policy = gen_policy(mdp)
    return mdp, policy


def _subprocess_collect_samples(args):
    itr, param_values, max_samples, queue = args
    global mdp
    global policy
    try:
        return _collect_samples(mdp, policy, itr, param_values, max_samples, queue)
    except Exception as e:
        import ipdb; ipdb.set_trace()

def rollout(mdp, policy, max_length=np.inf):
    states = []
    observations = []
    actions = []
    rewards = []
    pdists = []
    s, o = mdp.reset()
    path_length = 0
    while path_length < max_length:
        path_length += 1
        a, pdist = policy.get_action(o)
        next_s, next_o, r, d = mdp.step(s, a)
        states.append(s)
        observations.append(o)
        rewards.append(r)
        actions.append(a)
        pdists.append(pdist)
        if d:
            break
        s, o = next_s, next_o
    return dict(
        states=np.vstack(states),
        observations=np.vstack(observations),
        actions=np.vstack(actions),
        rewards=np.vstack(rewards).reshape(-1),
        pdists=np.vstack(pdists)
    )

def _collect_samples(mdp, policy, itr, param_values, max_samples, queue=None):
    try:
        policy.set_param_values(param_values)
        n_samples = 0
        paths = []
        while n_samples < max_samples:
            path = rollout(mdp, policy, max_samples - n_samples)
            paths.append(path)
            n_samples += len(path["states"])
            if queue is not None:
                queue.put(('samples', len(path["states"])))
        return paths
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

class RolloutSampler(object):

    def __init__(self, buf=None, n_parallel=None, gen_mdp=None, gen_policy=None):
        if buf is not None:
            n_parallel, gen_mdp, gen_policy = pickle.loads(buf)
        self._n_parallel = n_parallel
        self._setup_called = False
        self._gen_mdp = gen_mdp
        self._gen_policy = gen_policy
        self._pool = None
        self._mdp = None
        self._policy = None
        self._buf = buf

    def _setup(self):
        if not self._setup_called:
            if self._n_parallel > 1:
                self._pool = MemmapingPool(self._n_parallel)
                if self._buf is not None:
                    self._pool.map(SafeFunction(_init_subprocess), [self._buf] * self._n_parallel)
                else:
                    self._pool.map(SafeFunction(_init_subprocess), [(self._gen_mdp, self._gen_policy)] * self._n_parallel)
            else:
                self._mdp, self._policy = _init_mdp_policy(self._gen_mdp, self._gen_policy)
            self._setup_called = True

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._n_parallel > 1:
            self._pool.close()
            self._pool.terminate()
            self._pool = None
        else:
            self._mdp = None
            self._policy = None
        self._setup_called = False

    def collect_samples(self, itr, param_values, max_samples):
        if not self._setup_called:
            if self._n_parallel > 1:
                raise ValueError('Must enclose RolloutSampler in a with clause')
            else:
                self._setup()
        if self._n_parallel > 1:
            manager = Manager()
            queue = manager.Queue()
            args = itr, param_values, max_samples / self._n_parallel, queue
            map_result = self._pool.map_async(_subprocess_collect_samples, [args] * self._n_parallel)
            n_samples = 0
            max_progress = 1000000
            cur_progress = 0
            pbar = pyprind.ProgBar(max_progress)
            while not map_result.ready():
                map_result.wait(0.1)
                while not queue.empty():
                    ret = queue.get_nowait()
                    if ret[0] == 'samples':
                        n_samples += ret[1]
                new_progress = n_samples * max_progress / max_samples
                if new_progress < max_progress:#cur_progress < max_progress:
                    pbar.update(new_progress - cur_progress)
                cur_progress = new_progress
            pbar.stop()
            try:
                result_list = map_result.get()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
            return sum(result_list, [])
        else:
            return _collect_samples(self._mdp, self._policy, itr, param_values, max_samples)

sampler = RolloutSampler

if __name__ == '__main__':
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    launch_sampler(RolloutSampler)
