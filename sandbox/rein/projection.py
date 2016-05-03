from sklearn import manifold
from sklearn.random_projection import SparseRandomProjection
from matplotlib import pyplot as plt
import numpy as np
n_components = 2
X1 = np.load('/home/rein/workspace_python/rllab/data/mc_obs_act_trpo.npy')
X2 = np.load('/home/rein/workspace_python/rllab/data/mc_obs_act_trpo_ex.npy')
print(X1.shape, X2.shape)

mean = np.mean(np.vstack((X1,X2)), axis=0)
sd = np.std(np.vstack((X1,X2)), axis=0)
print(mean, sd)
X1 = (X1 - mean) / sd
X2 = (X2 - mean) / sd

batch_size = 5000
start_itr = 0
end_itr = 200

X1 = X1[batch_size * start_itr:batch_size * 180, :2]
X2 = X2[batch_size * start_itr:batch_size * 40, :2]

print(X1.shape, X2.shape)

n_samples = 200000
rand_ind1 = np.random.choice(range(X1.shape[0]), n_samples)
rand_ind2 = np.random.choice(range(X2.shape[0]), n_samples)

X1 = X1[rand_ind1, :]
X2 = X2[rand_ind2, :]

X = np.vstack((X1, X2))
# tsne = manifold.TSNE(
#     n_components=n_components, init='pca', random_state=0)
# Y = tsne.fit_transform(X)
# rp = SparseRandomProjection(n_components)
# rpf = rp.fit(X)
# Y1 = rpf.transform(X1)
# Y2 = rpf.transform(X2)
# Y = rpf.transform(X)
Y = X
Y1 = X1
Y2 = X2
color = [
    'blue' if i <= n_samples else 'red' for i in xrange(Y.shape[0])]
plt.scatter(
    Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, lw=0, alpha=0.05)
plt.show()

heatmap, xedges, yedges = np.histogram2d(Y1[:, 0], Y1[:, 1], bins=(256, 256))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
extent = [0,4, 0, 4]

plt.clf()
plt.imshow(heatmap, extent=extent, cmap="jet")
plt.show()

heatmap, xedges, yedges = np.histogram2d(Y2[:, 0], Y2[:, 1], bins=(256, 256))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap, extent=extent, cmap="jet")
plt.show()