from __future__ import print_function
from __future__ import absolute_import

import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridPlot(object):
    def __init__(self, grid_size, title=None):
        fig, ax = plt.subplots()  #
        #  = plt.figure(figsize=(5, 5))
        # ax = plt.axes(aspect=1)

        plt.tick_params(axis='x', labelbottom='off')
        plt.tick_params(axis='y', labelleft='off')
        if title:
            plt.title(title)

        self.grid_size = grid_size
        self.ax = ax
        self.figure = fig
        self.reset_grid()

    def reset_grid(self):
        self.ax.clear()
        self.ax.set_xticks(range(self.grid_size + 1))
        self.ax.set_yticks(range(self.grid_size + 1))
        self.ax.grid(True, linestyle='-', color=(0, 0, 0), alpha=1, linewidth=1)

    def add_text(self, x, y, text, gravity='center', size=10):
        # transform the grid index to coordinate
        x, y = y, self.grid_size - 1 - x
        if gravity == 'center':
            self.ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', size=size)
        elif gravity == 'left':
            self.ax.text(x + 0.05, y + 0.5, text, ha='left', va='center', size=size)
        elif gravity == 'top':
            self.ax.text(x + 0.5, y + 1 - 0.05, text, ha='center', va='top', size=size)
        elif gravity == 'right':
            self.ax.text(x + 1 - 0.05, y + 0.5, text, ha='right', va='center', size=size)
        elif gravity == 'bottom':
            self.ax.text(x + 0.5, y + 0.05, text, ha='center', va='bottom', size=size)
        else:
            raise NotImplementedError

    def color_grid(self, x, y, color, alpha=1.):
        x, y = y, self.grid_size - 1 - x
        self.ax.add_patch(patches.Rectangle(
            (x, y),
            1,
            1,
            facecolor=color,
            alpha=alpha
        ))
