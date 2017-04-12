import tempfile
import math
import gc

import numpy as np
import scipy.misc

from sandbox.young_clgan.goal.evaluator import evaluate_goals, convert_label
from sandbox.young_clgan.envs.base import FixedGoalGenerator

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import rc

# rc('text', usetex=True)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot_policy_reward(policy, env, limit, horizon=200, max_reward=6000, fname=None, grid_size=60,
                       return_rewards=False):
    """
    Complete evaluation of the policy to reach all points in a 2D grid
    :param limit: in a 2D square of this side-length
    :param grid_size: compute the difficulty of reaching every of these grid points
    :param horizon: in this many steps
    :param max_reward: should be high enough to mean it has reached the goal for several steps (just for plot)
    :param fname: where to save the pcolormesh
    :return also return the image
    """
    x, y = np.meshgrid(np.linspace(-limit, limit, grid_size), np.linspace(-limit, limit, grid_size))
    grid_shape = x.shape
    goals = np.hstack([
        x.flatten().reshape(-1, 1),
        y.flatten().reshape(-1, 1)
    ])
    z = evaluate_goals(goals, env, policy, horizon, 1)  # try out every goal in the grid
    print("Min return: {}\nMax return: {}\nMean return: {}".format(np.min(z), np.max(z), np.mean(z)))

    z = z.reshape(grid_shape)
    plt.figure()
    plt.clf()
    plt.pcolormesh(x, y, z, vmin=0, vmax=max_reward)
    plt.colorbar()
    if fname is not None:
        plt.savefig(fname, format='png')
        if return_rewards:
            return scipy.misc.imread(fname), z
        else:
            return scipy.misc.imread(fname)
    else:
        fp = tempfile.TemporaryFile()
        plt.savefig(fp, format='png')
        fp.seek(0)
        img = scipy.misc.imread(fp)
        fp.close()
        if return_rewards:
            return img, z
        else:
            return img


def save_image(fig=None, fname=None):
    if fname is None:
        fname = tempfile.TemporaryFile()
    if fig is not None:
        fig.savefig(fname)
    else:
        plt.savefig(fname, format='png')
    plt.close('all')
    fname.seek(0)
    img = scipy.misc.imread(fname)
    fname.close()
    return img


def plot_labeled_samples(samples, sample_classes=None, text_labels=None, markers=None, fname=None, limit=None,
                         center=None,
                         size=1000, colors=('k', 'm', 'c', 'y', 'r', 'g', 'b'), bounds=None):
    size = min(size, samples.shape[0])
    indices = np.random.choice(samples.shape[0], size, replace=False)
    samples = samples[indices, :]
    sample_classes = sample_classes[indices]
    if markers is None:
        markers = {i: 'o' for i in text_labels.keys()}

    unique_classes = list(set(sample_classes))
    assert (len(colors) > max(unique_classes))

    if len(samples[0]) >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in unique_classes:
            ax.scatter(
                samples[sample_classes == i, 0],
                samples[sample_classes == i, 1],
                samples[sample_classes == i, 2],
                # Choose a fixed color for each class.
                c=colors[i],
                marker=markers[i],
                alpha=0.8,
                label=text_labels[i]
            )
        if bounds is not None:
            plot_bounds(ax, bounds, dim=3)
        if limit:
            if center is None:
                center = np.zeros(2)
            ax.set_ylim3d(center[0] - limit, center[0] + limit)
            ax.set_xlim3d(center[1] - limit, center[1] + limit)
            ax.set_zlim3d(center[2] - limit, center[2] + limit)
    else:
        fig, ax = plt.subplots()
        for i in unique_classes:
            ax.scatter(
                samples[sample_classes == i, 0],
                samples[sample_classes == i, 1],
                # samples[sample_classes == i, 2],
                # Choose a fixed color for each class.
                c=colors[i],
                alpha=0.8,
                marker=markers[i],
                label=text_labels[i]
            )
        if bounds is not None:
            plot_bounds(ax, bounds, 2)
        if limit is not None:
            if center is None:
                center = np.zeros(2)
            ax.set_ylim(center[0] - limit, center[0] + limit)
            ax.set_xlim(center[1] - limit, center[1] + limit)
    # Place the legend to the right of the plot.
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if fname is not None:
        plt.savefig(fname, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plt.cla()
        # plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()
        return scipy.misc.imread(fname)
    else:
        fp = tempfile.TemporaryFile()
        plt.savefig(fp, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        fp.seek(0)
        img = scipy.misc.imread(fp)
        fp.close()
        # plt.cla()
        # plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()
        return img


def plot_bounds(ax, bounds, dim=2):
    if dim == 2:
        low, high = bounds
        i = 0
        a = np.copy(high)
        a[i] = low[i]
        b = np.copy(low)
        b[i] = high[i]
        ax.plot(*zip(high, a), color="b", label='state bounds')
        ax.plot(*zip(high, b), color="b")
        ax.plot(*zip(low, a), color="b")
        ax.plot(*zip(low, b), color="b")
    elif dim == 3:
        high, low = (np.array(b) for b in bounds)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            a = np.copy(high)
            a[i] = low[i]
            b = np.copy(low)
            b[j] = high[j]
            c = np.copy(low)
            c[k] = high[k]
            ax.plot(*zip(high[:3], a[:3]), color="b")
            ax.plot(*zip(low[:3], b[:3]), color="b")
            ax.plot(*zip(a[:3], b[:3]), color='b')
            ax.plot(*zip(a[:3], c[:3]), color='b')


def plot_gan_samples(gan, limit, fname=None, size=500):
    """Scatter size samples of the gan: no evaluation"""
    samples, _ = gan.sample_states(size)
    fig = plt.figure()
    # plt.clf()
    if np.size(samples[0]) >= 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
        ax.set_ylim3d(-limit, limit)
        ax.set_xlim3d(-limit, limit)
        ax.set_zlim3d(-limit, limit)
    else:
        plt.scatter(samples[:, 0], samples[:, 1])
        plt.ylim(-limit, limit)
        plt.xlim(-limit, limit)
    if fname is not None:
        plt.savefig(fname, format='png')
        # plt.cla()
        # plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()
        return scipy.misc.imread(fname)
    else:
        fp = tempfile.TemporaryFile()
        plt.savefig(fp, format='png')
        fp.seek(0)
        img = scipy.misc.imread(fp)
        fp.close()
        # plt.cla()
        # plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()
        return img


def plot_line_graph(fname=None, *args, **kwargs):
    plt.figure()
    plt.clf()
    plt.plot(*args, **kwargs)
    if fname is not None:
        plt.savefig(fname, format='png')
        return scipy.misc.imread(fname)
    else:
        fp = tempfile.TemporaryFile()
        plt.savefig(fp, format='png')
        fp.seek(0)
        img = scipy.misc.imread(fp)
        fp.close()
        return img
