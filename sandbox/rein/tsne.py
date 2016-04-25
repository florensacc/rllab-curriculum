from sklearn import manifold
from sklearn.random_projection import SparseRandomProjection
from matplotlib import pyplot as plt
import numpy as np
n_components = 2
X1 = np.load('/home/rein/workspace_python/rllab/data/mc_obs_act_trpo.npy')
X2 = np.load('/home/rein/workspace_python/rllab/data/trpo_obs_act_ex.npy')
print(X1.shape, X2.shape)

batch_size = 5000
start_itr = 0
end_itr = 200

X1 = X1[batch_size * start_itr:batch_size * end_itr, :]
X2 = X2[batch_size * start_itr:batch_size * end_itr, :]

n_samples = 2500
rand_ind1 = np.random.choice(range(X1.shape[0]), n_samples)
rand_ind2 = np.random.choice(range(X2.shape[0]), n_samples)

X1 = X1[rand_ind1, :]
X2 = X2[rand_ind2, :]

X = np.vstack((X1, X2))
tsne = manifold.TSNE(
    n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
# rp = SparseRandomProjection(n_components)
# Y = rp.fit(X).transform(X)
# Y = X
color = [
    'blue' if i <= n_samples else 'red' for i in xrange(Y.shape[0])]
label = [
    'expl' if i <= n_samples else 'no expl' for i in xrange(Y.shape[0])]
plt.scatter(
    Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, lw=0, alpha=0.5, label=label)
plt.show()
