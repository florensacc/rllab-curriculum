import tempfile
import math

import numpy as np
import scipy.misc

from sandbox.young_clgan.lib.goal.evaluator import evaluate_goals, convert_label
from sandbox.young_clgan.lib.envs.base import FixedGoalGenerator

import matplotlib

matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


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


def plot_labeled_samples(samples, sample_classes=None, text_labels=None, markers=None, fname=None, limit=None,
                         size=1000, colors=('k', 'm', 'c', 'y', 'r', 'g', 'b')):
    size = min(size, samples.shape[0])
    indices = np.random.choice(samples.shape[0], size, replace=False)
    samples = samples[indices, :]
    sample_classes = sample_classes[indices]
    if markers is None:
        markers = {i:'o' for i in text_labels.keys()}

    unique_classes = list(set(sample_classes))
    assert (len(colors) > max(unique_classes))

    fig = plt.figure()
    if len(samples[0]) >= 3:
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
        ax.set_ylim3d(-limit, limit)
        ax.set_xlim3d(-limit, limit)
        ax.set_zlim3d(-limit, limit)
            # high, low = self.wrapped_env.observation_space.bounds
            # for i in range(3):
            #     j = (i + 1) % 3
            #     k = (i + 2) % 3
            #     a = high[idx]
            #     a[i] = low[idx[i]]
            #     b = low[idx]
            #     b[j] = high[idx[j]]
            #     c = low[idx]
            #     c[k] = high[idx[k]]
            #     # import pdb; pdb.set_trace()
            #     print("segments are: ", *zip(high[idx], b), 'and ', *zip(high[idx], b), 'and ', *zip(a, b), ' and ', *zip(a, c))
            #     ax.plot(*zip(high[idx], a), color="b")
            #     ax.plot(*zip(low[idx], b), color="b")
            #     ax.plot(*zip(a, b), color='b')
            #     ax.plot(*zip(a, c), color='b')
    else:
        for i in unique_classes:
            plt.scatter(
                samples[sample_classes == i, 0],
                samples[sample_classes == i, 1],
                # samples[sample_classes == i, 2],
                # Choose a fixed color for each class.
                c=colors[i],
                alpha=0.8,
                label=text_labels[i]
            )

        plt.ylim(-limit, limit)
        plt.xlim(-limit, limit)
    # Place the legend to the right of the plot.
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if fname is not None:
        plt.savefig(fname, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        return scipy.misc.imread(fname)
    else:
        fp = tempfile.TemporaryFile()
        plt.savefig(fp, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        fp.seek(0)
        img = scipy.misc.imread(fp)
        fp.close()
        return img


def plot_gan_samples(gan, limit, fname=None, size=500):
    """Scatter size samples of the gan: no evaluation"""
    samples, _ = gan.sample_goals(size)
    fig = plt.figure()
    # plt.clf()
    if np.size(samples[0]) >= 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
    else:
        plt.scatter(samples[:, 0], samples[:, 1])
    plt.ylim(-limit, limit)
    plt.xlim(-limit, limit)
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


def plot_line_graph(fname=None, *args, **kwargs):
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
