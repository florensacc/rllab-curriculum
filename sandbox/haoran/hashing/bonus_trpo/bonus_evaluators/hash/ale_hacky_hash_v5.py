"""
A more flexible version
- allows postprocessing the ram values (useful for e.g. changing the coordinate scales)
- conveniently include specific RAM entries without changing the class interface
"""

from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash import ale_ram_info
from collections import OrderedDict
import numpy as np
import multiprocessing as mp
import copy

class ALEHackyHashV5(Hash):
    def __init__(self,item_dim, game, ram_names, extra_info=dict(), parallel=False):
        assert item_dim == 128 # must be RAM
        self.item_dim = item_dim
        all_ram_info = getattr(ale_ram_info,game)
        self.ram_info = OrderedDict([
            (name, all_ram_info[name])
            for name in sorted(ram_names)
        ])
        for name, extra_info in extra_info.items():
            self.ram_info[name].update(extra_info)

        # determine the number of distinct values in each ram entry
        self.n_values = []
        for name, info in self.ram_info.items():
            if "grid_size" in info.keys():
                n_value = int(np.ceil(len(info["values"]) / info["grid_size"]))
            else:
                n_value = len(info["values"])
            self.n_values.append(n_value)

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

    def init_shared_dict(self, shared_dict):
        self.shared_dict = shared_dict


    def compute_keys(self, items):
        """
        Convert the tuple of relevant ram values into a single integer, exact like converting a binary number to a decimal integer.
        :param items: a N x 128 matrix of N ram states
        """
        # sometimes items have the shape (n_items, 128, 1), or (n_items, 1, 128, 1), like images
        if len(items.shape) > 2:
            items = np.asarray([item.ravel() for item in items],dtype=int)
        else:
            items = items.astype(int)
        n_items = items.shape[0]

        # convert useful ram values to an integer code
        # similar to converting a binary number to a decimal
        keys = np.zeros(n_items,dtype=int)
        for i,name in enumerate(self.ram_info.keys()):
            info = self.ram_info[name]
            index = info["index"]
            value_type = info["value_type"]
            values = info["values"]
            _items = items[:,index] # vector of a specific RAM entry
            if value_type == "range":
                # v -> v - v_min
                value_min = np.amin(values)
                _items_short = _items - value_min
                if "grid_size" in info.keys():
                    _items_short = np.floor(_items_short / info["grid_size"]).astype(int)
                if "random" in info.keys() and info["random"]:
                    _items_short = np.random.randint(
                        low=0,
                        high=self.n_values[i],
                        size=_items_short.shape
                    )
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
        :param keys: a vector of integer keys
        """
        assert len(keys.shape) == 1
        n_keys = len(keys)
        values = np.zeros((n_keys,len(self.n_values)),dtype=int)
        remains = keys
        for i,name in enumerate(self.ram_info.keys()):
            s = int(np.prod(self.n_values[i+1:]))
            _values_short = remains // s

            if original_range:
                info = self.ram_info[name]
                index = info["index"]
                value_type = info["value_type"]
                ram_values = info["values"]
                if value_type == "range":
                    value_min = np.amin(ram_values)
                    if "grid_size" in info.keys():
                        _values = _values_short * info["grid_size"] + value_min
                    else:
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

    def total_state_count(self):
        if self.parallel:
            return np.count_nonzero(self.table)
        else:
            try:
                return len(self.table.keys())
            except:
                return np.count_nonzero(self.table)
