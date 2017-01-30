#!/usr/bin/env python

import argparse
import h5py
import matplotlib.pyplot as plt

COLORS = ['m', 'c', 'y', 'b', 'r']
MARKERS = ['+', 'x', 'd', 'o', 's']
SCATTER = False

def plot_returns(h5_file, groups, x_key, y_key, screen_print=False):
    # Structure of h5_file should be:
    #     y = f[group][x][y_key][()]
    # Note that group can be '/l-inf/fs3/' for example, if there are two or
    # more levels of groups in h5 file

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)

    f = h5py.File(h5_file, 'r')
    axes = []
    for i,g in enumerate(groups):
        points = []
        for x in f[g].keys():
            points.append((float(x), f[g][x][y_key][()]))

        points = sorted(points, key=lambda x: x[0])
        if SCATTER:
            a = plt.scatter([p[0] for p in points], [p[1] for p in points], \
                            c=COLORS[i], s=20, marker=MARKERS[i])
        else:
            a, = plt.plot([p[0] for p in points], [p[1] for p in points], \
                         MARKERS[i]+COLORS[i]+'-', ms=6)
        axes.append(a)

        if screen_print:
            print("Group:", g)
            for p in points:
                print("\t", p[0], "\t", p[1])

    if SCATTER:
        plt.legend(axes, groups, scatterpoints=1, loc='lower left', \
                   ncol=3, fontsize=14)
    else:
        plt.legend(axes, groups, loc='lower left', ncol=3, fontsize=14)

    plt.title('Pong with FGSM Adversary', fontsize=14)
    plt.xlabel(x_key, fontsize=14)
    plt.ylabel(y_key, fontsize=14)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    plot_returns(args.returns_h5, ['l-inf', 'l1', 'l2'], 'fgsm_eps', 'avg_return', screen_print=True)


if __name__ == "__main__":
    main()
