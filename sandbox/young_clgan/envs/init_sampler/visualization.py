import tempfile

import matplotlib
import numpy as np
import scipy.misc
from sandbox.young_clgan.goal.evaluator import evaluate_goals, convert_label

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_policy_reward(policy, env, limit, horizon=200, max_reward=6000, fname=None, grid_size=60):
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
        return scipy.misc.imread(fname)
    else:
        fp = tempfile.TemporaryFile()
        plt.savefig(fp, format='png')
        fp.seek(0)
        img = scipy.misc.imread(fp)
        fp.close()
        return img


def plot_labeled_samples(samples, labels, limit,
                         fname=None, size=1000, colors=['red', 'green', 'blue', 'yellow']):
    sample_classes, text_labels = convert_label(labels)
    size = min(size, samples.shape[0])
    indices = np.random.choice(samples.shape[0], size, replace=False)
    samples = samples[indices, :]
    sample_classes = sample_classes[indices]

    # point_class = labels[:, 0] * 2 + labels[:, 1]
    plt.figure()
    plt.clf()
    # text_labels = ['', 'Reward not sensible', 'Goal already accomplished', 'Desired goals']

    # colors = ['b', 'g', 'r', 'y', 'c', 'm']
    colors = ['k', 'm', 'c', 'y', 'r', 'g', 'b']

    unique_classes = list(set(sample_classes))

    assert (len(colors) > max(unique_classes))

    for i in unique_classes:
        plt.scatter(
            samples[sample_classes == i, 0],
            samples[sample_classes == i, 1],
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
    plt.figure()
    plt.clf()
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
