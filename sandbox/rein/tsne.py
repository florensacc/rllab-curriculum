from sklearn import manifold
from matplotlib import pyplot as plt
import numpy as np
n_components = 2
X = np.load('/home/rein/workspace_python/rllab/data/obs_act.npy')
n_samples = 5000
rand_ind = np.random.choice(range(X.shape[0]), n_samples)
tsne = manifold.TSNE(
    n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X[rand_ind, :])
color = [
    'blue' if i > n_samples / 2 else 'red' for i in xrange(Y.shape[0])]
plt.scatter(
    Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, lw=0)
plt.show()