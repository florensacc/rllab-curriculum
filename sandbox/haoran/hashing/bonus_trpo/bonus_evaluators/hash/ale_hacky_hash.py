from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
import numpy as np

class ALEHackyHash(Hash):
    def __init__(self,item_dim, game):
        """
        Hand-designed state encodings using RAM
        """
        raise NotImplementedError # see v2 for changes to montezuma_revenge
        assert item_dim == 128 # must be RAM
        self.item_dim = item_dim

        if game == "montezuma_revenge":
            self.ram_info = {
                27: dict(
                    name="beam_wall",
                    values=[253,209],
                    meanings=["off","on"]
                ),
                42: dict(
                    name="x",
                    values=range(255),
                ),
                43: dict(
                    name="y",
                    values=range(255),
                ),
                51: dict(
                    name="room_number",
                    values=range(255),
                ),
                # 83: dict  (
                #     name="beam_countdown",
                #     values=range(37)
                # )
            }
        else:
            raise NotImplementedError

        self.ram_indices = sorted(list(self.ram_info.keys())) # fix the ordering
        self.ram_names = [self.ram_info[index] for index in self.ram_indices]
        n_values = [
            len(self.ram_info[index]["values"])
            for index in self.ram_indices
        ]
        # ndarray, one dim for one ram index
        self.table = np.zeros(tuple(n_values),dtype=int)

    def compute_keys(self, items):
        if len(items.shape) == 3:
            items = [item.ravel() for item in items]
        keys = [
            tuple([
                self.ram_info[index]["values"].index(item[index])
                for index in self.ram_indices
            ])
            for item in items
        ]
        return keys

    def inc_keys(self, keys):
        for key in keys:
            self.table[key] += 1

    def query_keys(self, keys):
        counts = [self.table[key] for key in keys]
        return counts

    def reset(self):
        self.table = np.zeros_like(self.table)
