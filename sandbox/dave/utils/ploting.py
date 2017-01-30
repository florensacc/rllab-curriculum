import numpy as np
from matplotlib import pyplot as plt
import collections
import gc
from functools import reduce
from rllab.misc import logger
from os import path as osp
from pylab import *

def plot_heatmap(paths, type, prefix=''):
    # fig, ax = plt.subplots()
    import pdb; pdb.set_trace()
    x_goal = [path["env_infos"]["x_goal"] for path in paths]
    xsg = np.array([x[0] for x in x_goal])
    n = np.round(np.sqrt(len(xsg)))
    xsg = np.reshape(xsg, (n,n))
    y_goal = [path["env_infos"]["y_goal"] for path in paths]
    ysg = np.array([y[0] for y in y_goal])
    ysg = np.reshape(ysg, (n,n))

    x_lego = [path["env_infos"]["x_lego"] for path in paths]
    xsl = np.array([x[0] for x in x_lego])
    xsl = np.reshape(xsl, (n,n))
    y_lego = [path["env_infos"]["y_lego"] for path in paths]
    ysl = np.array([y[0] for y in y_lego])
    ysl = np.reshape(ysl, (n,n))
    distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
    zs = [d[-1] for d in distances_to_goal]
    zs = np.reshape(zs, (n,n))


    xs = xsg - xsl
    ys = ysg - ysl

    zmin = 0
    zmax = 0.3
    zs[(zs<zmin) | (zs>zmax)] = None

    # Create the contour plot
    CS = plt.contourf(xs, ys, zs, 15, cmap=plt.cm.rainbow,
                      vmax=zmax, vmin=zmin)
    plt.colorbar()
    plt.show()

    #
    # plt.cla()
    # plt.clf()
    # plt.close('all')
    # gc.collect()


    log_dir = logger.get_snapshot_dir()


    plt.savefig(osp.join(log_dir, prefix + 'heatmap_ '+ type + '.png'))
    # plt.close()


def plot_finaldistance_hist(paths, furthest, prefix=''):

    distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
    zs = [d[-1] for d in distances_to_goal]

    zs_mean = np.mean(zs)
    zs_std = np.std(zs)
    zs_min = np.min(zs)
    zs_max = np.max(zs)

    plt.hist(zs)
    plt.title("Final distance object-goal Histogram")
    plt.xlabel("Final distance (m)")
    plt.ylabel("Frequency")

    fig = plt.gcf()

    plt.show()

    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()


    log_dir = logger.get_snapshot_dir()
    plt.savefig(osp.join(log_dir, prefix + 'histogram.png'))
    plt.close()

def plot_scatter_heatmap(paths, type, prefix=''):
    fig, ax = plt.subplots()

    x_goal = [path["env_infos"]["x_goal"] for path in paths]
    xsg = np.array([x[0] for x in x_goal])
    y_goal = [path["env_infos"]["y_goal"] for path in paths]
    ysg = np.array([y[0] for y in y_goal])

    x_lego = [path["env_infos"]["x_lego"] for path in paths]
    xsl = np.array([x[0] for x in x_lego])
    y_lego = [path["env_infos"]["y_lego"] for path in paths]
    ysl = np.array([y[0] for y in y_lego])
    distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]

    xs = xsg - xsl
    ys = ysg - ysl
    zs = [d[-1] for d in distances_to_goal]

    furthest = np.max(np.maximum(abs(xs), abs(ys)))

    colmap = cm.ScalarMappable(cmap=cm.CMRmap)
    colmap.set_array(zs)

    yg = ax.scatter(xs, ys, c=cm.CMRmap(zs / max(zs)))
    cb = fig.colorbar(colmap)

    ax.set_title(prefix + 'Final distance object - goal (m)')
    ax.set_xlabel('distance to object (m)')
    ax.set_ylabel('distance to object (m)')

    ax.set_xlim([-furthest, furthest])
    ax.set_ylim([-furthest, furthest])

    plt.show()
    #
    # plt.cla()
    # plt.clf()
    # plt.close('all')
    # gc.collect()


    log_dir = logger.get_snapshot_dir()


    plt.savefig(osp.join(log_dir, prefix + 'heatmap_ '+ type + '.png'))
    # plt.close()

