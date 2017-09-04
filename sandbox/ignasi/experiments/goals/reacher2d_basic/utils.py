import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sandbox.ignasi.state.evaluator import evaluate_states
from sandbox.ignasi.logging.visualization import save_image



def plot_policy_performance(policy, env, horizon, n_traj=5, grid_size=50, fname=None):
    limit = 0.2
    x, y = np.meshgrid(np.linspace(-limit, limit, grid_size), np.linspace(-limit, limit, grid_size))
    grid_shape = x.shape
    goals = np.hstack([
        x.flatten().reshape(-1, 1),
        y.flatten().reshape(-1, 1)
    ])
    z = evaluate_states(
        goals, env, policy, horizon, n_traj=n_traj, key='goal_reached',
        aggregator=(np.max, np.mean)
    )

    z = z.reshape(grid_shape)
    plt.figure()
    plt.clf()
    plt.pcolormesh(x, y, z, vmin=0.0, vmax=1.0)
    plt.colorbar()
    
    return save_image(fname=fname)
    
    
    
def plot_generator_samples(generator, size=100, fname=None):
    
    goals, _ = generator.sample_states(size)
    
    limit = 0.2
    
    
    goals_dim = 2
    
    plt.figure()
    plt.clf()
    plt.scatter(goals[:, 0], goals[:, 1], s=10)
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    
        
    
    img = save_image(fname=fname)
    return img

        
        
    
    
    
    