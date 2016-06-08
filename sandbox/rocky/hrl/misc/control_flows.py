from __future__ import print_function
from __future__ import absolute_import


class MultiAlgo(object):

    def __init__(self, algos_dict):
        self.algos_dict = algos_dict

    def train(self):
        keys = sorted(self.algos_dict.keys())
        for key in keys:
            self.algos_dict[key].train()
