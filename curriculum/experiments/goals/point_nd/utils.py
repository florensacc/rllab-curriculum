import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from curriculum.state.evaluator import evaluate_states
from curriculum.logging.visualization import save_image


def plot_policy_performance(policy, env, horizon, n_samples=200, n_traj=10, fname=None):
    goals_dim = env.dim
    goals_bound = env.goal_bounds[:len(env.goal_bounds) // 2]
    
    goals = np.random.uniform(-goals_bound, goals_bound, [n_samples, goals_dim])
    
    limit = np.max(goals_bound)
    
    success_rates = evaluate_states(
        goals, env, policy, horizon, n_traj=n_traj, key='goal_reached',
        aggregator=(np.max, np.mean)
    )
    
    plt.clf()
    
    if goals_dim == 2:
        plt.scatter(goals[:, 0], goals[:, 1], c=success_rates, s=10, cmap='plasma')
        plt.colorbar()
        plt.axis('equal')
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        p = ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c=success_rates, s=10, cmap='plasma')
        fig.colorbar(p)
        ax.axis('equal')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    img = save_image(fname=fname)
    
    return img
    
    
def plot_generator_samples(generator, env=None, size=100, fname=None):
    
    goals, _ = generator.sample_states(size)
    
    if env is None:
        limit = np.max(np.abs(goals))
    else:
        goals_bound = env.goal_bounds[:len(env.goal_bounds) // 2]
        limit = np.max(goals_bound)
    
    
    goals_dim = goals.shape[1]
    
    if goals_dim == 2:
        plt.scatter(goals[:, 0], goals[:, 1], s=10)
        plt.axis('equal')
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        p = ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], s=10)
        ax.axis('equal')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    img = save_image(fname=fname)
    return img

        
        
    
    
    
    