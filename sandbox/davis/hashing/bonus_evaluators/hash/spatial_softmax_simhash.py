from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash import ale_ram_info
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.sim_hash_v2 import SimHashV2
# from sandbox.davis.hashing.bonus_evaluators.hash.ale_hacky_hash_v5 import ALEHackyHashV5

from collections import OrderedDict
import numpy as np
import copy


class SimHackyHash(SimHashV2):
    def __init__(
            self,
            item_dim,
            game,
            ram_names,
            use_onehot_encoding=False,
            extra_info=dict(),
            dim_key=128,
            bucket_sizes=None,
            parallel=False):
        self.hacky_item_dim = item_dim
        self.init_hacky_hash(item_dim, game, ram_names, extra_info, parallel)
        self.use_onehot_encoding = use_onehot_encoding
        if use_onehot_encoding:
            self.simhash_item_dim = sum(self.n_values)
        else:
            self.simhash_item_dim = len(self.ram_info)
        super(SimHackyHash, self).__init__(self.simhash_item_dim, dim_key, bucket_sizes, parallel)
        self.item_dim = self.hacky_item_dim

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        state = dict()
        for k, v in iter(self.__dict__.items()):
            if k not in self.unpicklable_list:
                state[k] = v
            elif k in self.snapshot_list:
                state[k] = copy.deepcopy(v)
        return state

    def compute_keys(self, items):
        """
        Compute the keys for many items (row-wise stacked as a matrix)
        """
        hacky_values = self.compute_hacky_values(items)
        import pdb; pdb.set_trace()
        if self.use_onehot_encoding:
            simhash_items = self.onehot_encode(hacky_values)
        else:
            simhash_items = hacky_values
        return super(SimHackyHash, self).compute_keys(simhash_items)

    def onehot_encode(self, hacky_values):
        encoding = np.zeros((len(hacky_values), self.simhash_item_dim))
        reference_index = 0
        for i, info in enumerate(self.ram_info.values()):
            hacky_value = hacky_values[:, i]
            onehot_index = reference_index + hacky_value.astype(int)
            encoding[:, onehot_index] = 1
            reference_index += self.n_values[i]
        return encoding

    def compute_hacky_values(self, items):
        if len(items.shape) > 2:
            items = np.asarray([item.ravel() for item in items], dtype=int)
        else:
            items = items.astype(int)
        n_items, _ = items.shape
        n_dims = len(self.n_values)

        values = np.zeros((n_items, n_dims))
        for i, info in enumerate(self.ram_info.values()):
            index = info["index"]
            entry = items[:, index]
            if info["value_type"] == "range":
                offset = entry - info["values"][0]
                offset = coerce_to_bounds(offset, 0, info["values"][-1])
                if "grid_size" in info.keys():
                    offset = np.floor(offset / info["grid_size"]).astype(int)
            elif info["value_type"] == "categorical":
                offset = np.where(entry.reshape(-1, 1) == info["values"])[1]
            else:
                raise NotImplementedError
            values[:, i] = offset
        return values

    def init_hacky_hash(self, item_dim, game, ram_names, extra_info, parallel):
        assert item_dim == 128  # must be RAM
        self.item_dim = item_dim
        all_ram_info = getattr(ale_ram_info, game)
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
        self.parallel = parallel

    def total_state_count(self):
        return 0  # not yet implemented


def get_index_offset(hacky_value, info):
    if info["value_type"] == "range":
        offset = hacky_value - info["values"][0]
        offset = coerce_to_bounds(offset, 0, info["values"][-1])
        if "grid_size" in info.keys():
            offset = np.floor(offset / info["grid_size"]).astype(int)
        return offset
    elif info["value_type"] == "categorical":
        return np.where(hacky_value.reshape(-1, 1) == info["values"])[1]
    else:
        raise NotImplementedError


def coerce_to_bounds(item, lower, upper):
    return np.minimum(upper, np.maximum(item, lower))
