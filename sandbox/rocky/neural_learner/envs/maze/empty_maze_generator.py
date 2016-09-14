import numpy as np

class EmptyMazeGenerator(object):

    def __init__(self):
        pass

    def gen_maze(self, n_row, n_col):
        return np.ones((n_row, n_col))

