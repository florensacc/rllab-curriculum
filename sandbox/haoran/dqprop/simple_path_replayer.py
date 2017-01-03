from collections import deque

class SimplePathReplayer(object):
    """
    Simply records paths in a FIFO manner.
    All recorded paths are replayed.
    """
    def __init__(
        self,
        max_pool_size,
        min_pool_size=0,
    ):
        assert max_pool_size > 0
        self._max_pool_size = max_pool_size
        self._min_pool_size = min_pool_size
        self._pool = deque()
        self._current_pool_size = 0

    def record_paths(self, paths):
        for path in paths:
            self._pool.append(path)
            path_len = len(path["rewards"])
            self._current_pool_size += path_len

        while self._current_pool_size > self._max_pool_size:
            path = self._pool.popleft()
            path_len = len(path["rewards"])
            self._current_pool_size -= path_len

    def replay_paths(self):
        if self._current_pool_size > self._min_pool_size:
            return list(self._pool)
        else:
            return []

    @property
    def pool_size(self):
        return self._current_pool_size
