"""
Implement the hash table as a vector instead of a multi-dimensional array, so that computing keys can be vectorized.
"""

from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
import numpy as np
import multiprocessing as mp
import copy

class ALEHackyHashV4(Hash):
    def __init__(self,item_dim, game, parallel=False):
        """
        This hash is taylored for clearing the first room (take key and go to room number 2)
        """
        assert item_dim == 128 # must be RAM
        self.item_dim = item_dim

        if game == "montezuma_revenge":
            self.ram_info = {
                3: dict(
                    name="room",
                    values=range(24),
                    value_type="range",
                ),
                42: dict(
                    name="x",
                    values=range(0,152),
                    value_type="range",
                ),
                43: dict(
                    name="y",
                    values=range(148,256),
                    value_type="range",
                ),
                47: dict(
                    name="skeleton_location",
                    values=range(20,80), # not exactly the min/max, but good enough
                    value_type="range",
                ),
                67: dict(
                    name="objects", # 1st level: doors, skeleton, key
                    values=range(16,32),
                    value_type="range",
                ),
            }
        else:
            raise NotImplementedError

        self.ram_indices = sorted(list(self.ram_info.keys())) # fix the ordering
        self.ram_names = [self.ram_info[index] for index in self.ram_indices]
        self.n_values = [
            len(self.ram_info[index]["values"])
            for index in self.ram_indices
        ]
        if parallel:
            self.table_lock = mp.Value('i')
            self.table = np.frombuffer(
                mp.RawArray('i', int(np.prod(self.n_values))),
                np.int32,
            )
            self.unpicklable_list = ["table_lock","table"]
            self.snapshot_list = ["table"]
            self.rank = None
        else:
            self.table = dict()
            self.unpicklable_list = []
            self.snapshot_list = []
        self.parallel = parallel

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        state = dict()
        for k,v in iter(self.__dict__.items()):
            if k not in self.unpicklable_list:
                state[k] = v
            elif k in self.snapshot_list:
                state[k] = copy.deepcopy(v)
        return state

    def init_rank(self,rank):
        self.rank = rank


    def compute_keys(self, items):
        # sometimes items have the shape (n_items, 128, 1), like images
        if len(items.shape) > 2:
            items = np.asarray([item.ravel() for item in items],dtype=int)
        else:
            items = items.astype(int)
        n_items = items.shape[0]

        # convert useful ram values to an integer code
        keys = np.zeros(n_items,dtype=int)
        for i in range(len(self.ram_indices)):
            index = self.ram_indices[i]
            _items = items[:,index]
            value_type = self.ram_info[index]["value_type"]
            values = self.ram_info[index]["values"]

            if value_type == "range":
                # v -> v - v_min
                value_min = np.amin(values)
                _items_short = _items - value_min
            elif value_type == "categorical":
                # v -> index of v
                _items_short = np.zeros_like(_items)
                for j in range(len(values)):
                    _items_short += j * (_items == values[j])
            else:
                raise NotImplementedError
            keys +=  _items_short * np.prod(self.n_values[i+1:]).astype(int)
        return keys

    def keys_to_values(self, keys, original_range=True):
        """
        Convert the integer key to important ram values
        """
        assert len(keys.shape) == 1
        n_keys = len(keys)
        values = np.zeros((n_keys,len(self.ram_indices)),dtype=int)
        remains = keys
        for i in range(len(self.ram_indices)):
            s = int(np.prod(self.n_values[i+1:]))
            _values_short = remains // s

            if original_range:
                index = self.ram_indices[i]
                value_type = self.ram_info[index]["value_type"]
                ram_values = self.ram_info[index]["values"]
                if value_type == "range":
                    value_min = np.amin(ram_values)
                    _values = _values_short + value_min
                elif value_type == "categorical":
                    _values = np.zeros_like(_values_short,dtype=int)
                    for j in range(len(ram_values)):
                        _values += ram_values[j] * (_values_short == j)
                else:
                    raise NotImplementedError
                values[:,i] = _values
            else:
                values[:,i] = _values_short
            remains = remains % s
        return values

    def inc_keys(self, keys):
        if self.parallel:
            with self.table_lock.get_lock():
                np.add.at(self.table, list(keys), 1)
        else:
            for key in keys:
                if key not in self.table:
                    self.table[key] = 1
                else:
                    self.table[key] += 1



    def query_keys(self, keys):
        if self.parallel:
            counts = self.table[list(keys)]
        else:
            counts = []
            for key in keys:
                if key not in self.table:
                    counts.append(0)
                else:
                    counts.append(self.table[key])
        return counts

    def reset(self):
        if self.parallel:
            with self.table_lock.get_lock():
                self.table = np.zeros_like(self.table)
        else:
            self.table = dict()
