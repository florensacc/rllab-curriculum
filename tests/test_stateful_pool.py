from __future__ import print_function
from __future__ import absolute_import
import itertools


def _worker_collect_once(_):
    return 'a', 1


def _worker_run_map(G):
    return G.worker_id


def _worker_run_each(G):
    return G.worker_id


def test_stateful_pool_run_map():
    from rllab.sampler import stateful_pool
    stateful_pool.singleton_pool.initialize(n_parallel=3)
    results = stateful_pool.singleton_pool.run_map(_worker_run_map, [tuple()] * 100)
    ids = itertools.groupby(sorted(results), key=lambda x: x)
    ids = dict(map(lambda x: (x[0], list(x[1])), ids))
    for idx in xrange(3):
        assert len(ids[idx]) > 0
    assert len(results) == 100


def test_stateful_pool_run_each():
    from rllab.sampler import stateful_pool
    stateful_pool.singleton_pool.initialize(n_parallel=3)
    results = stateful_pool.singleton_pool.run_each(_worker_run_each, [tuple()] * 3)
    assert tuple(results) == (0, 1, 2)


def test_stateful_pool_collect():
    from rllab.sampler import stateful_pool
    stateful_pool.singleton_pool.initialize(n_parallel=3)
    results = stateful_pool.singleton_pool.run_collect(_worker_collect_once, 3, show_prog_bar=False)
    assert len(results) >= 3


def test_stateful_pool_over_capacity():
    from rllab.sampler import stateful_pool
    stateful_pool.singleton_pool.initialize(n_parallel=4)
    results = stateful_pool.singleton_pool.run_collect(_worker_collect_once, 3, show_prog_bar=False)
    assert len(results) >= 3
