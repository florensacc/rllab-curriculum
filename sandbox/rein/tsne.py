from sklearn import manifold
from sklearn.random_projection import SparseRandomProjection
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import numpy as np
n_components = 2
_X1 = np.load('/home/rein/workspace_python/rllab/data/mc_obs_act_trpo.npy')
_X2 = np.load('/home/rein/workspace_python/rllab/data/mc_obs_act_trpo_ex.npy')
print(_X1.shape, _X2.shape)

batch_size = 5000
start_itr = 0
end_itr = 200

X1 = _X1[batch_size * start_itr:batch_size * end_itr, :]
X2 = _X2[batch_size * start_itr:batch_size * end_itr, :]

n_samples = 25000
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
# X1 = rpf.transform(X1)
# X2 = rpf.transform(X2)
color = [
    'blue' if i <= n_samples else 'red' for i in xrange(X1.shape[0])]

from mpl_toolkits.mplot3d import Axes3D

# f, ax = plt.subplots(2)
f = plt.figure()
ax1 = f.add_subplot(211, projection='3d')
ax2 = f.add_subplot(212, projection='3d')
scatter1 = ax1.scatter(
    X1[:, 0], X1[:, 1], X1[:, 2], c=color, cmap=plt.cm.Spectral, lw=0, alpha=0.05)
scatter2 = ax2.scatter(
    X2[:, 0], X2[:, 1], X2[:, 2], c=color, cmap=plt.cm.Spectral, lw=0, alpha=0.05)
scatter1.set_offsets(X1)
scatter2.set_offsets(X2)
# plt.show()
def animate(i):
#     end_itr = 200
#     print(i)
#     X1 = _X1[batch_size * start_itr:batch_size * end_itr, :]
#     X2 = _X2[batch_size * start_itr:batch_size * end_itr, :]
# 
#     n_samples = 25000
#     rand_ind1 = np.random.choice(range(X1.shape[0]), n_samples)
#     rand_ind2 = np.random.choice(range(X2.shape[0]), n_samples)
# 
#     X1 = X1[rand_ind1, :]
#     X2 = X2[rand_ind2, :]
# #     X = np.vstack((X1, X2))
# #     rp = SparseRandomProjection(n_components)
# #     rpf = rp.fit(X)
# #     Y = rpf.transform(X)
    scatter1._offsets3d=( np.ma.ravel(X1[:,0]) , np.ma.ravel(X1[:,1]) , np.ma.ravel(X1[:,2]) )
#     scatter1.set_offsets(X1)
#     scatter2.set_offsets(X2)

    return scatter1, scatter2


# Init only required for blitting to give a clean slate.
def init():
    scatter1.set_offsets(X1)
    scatter2.set_offsets(X2)
    return scatter1, scatter2


ani = animation.FuncAnimation(f, animate, np.arange(1, 200),
                              interval=25, blit=True)
plt.show()
