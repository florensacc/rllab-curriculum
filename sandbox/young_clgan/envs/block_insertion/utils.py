import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sandbox.young_clgan.state.evaluator import evaluate_states
from sandbox.young_clgan.logging.visualization import save_image



def plot_policy_performance(policy, env, horizon, n_samples=200, n_traj=5, fname=None):
    goals_dim = 3
    goal_lb = env.goal_lb
    goal_ub = env.goal_ub
    
    goals = np.random.uniform(goal_lb, goal_ub, [n_samples, goals_dim])
    
    success_rates = evaluate_states(
        goals, env, policy, horizon, n_traj=n_traj, key='goal_reached',
        aggregator=(np.max, np.mean)
    )
    
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c=success_rates, s=10, cmap=pylab.cm.jet)  # 'plasma')
    fig.colorbar(p)
    ax.set_xlim(goal_lb[0], goal_ub[0])
    ax.set_ylim(goal_lb[1], goal_ub[1])
    ax.set_zlim(goal_lb[2], goal_ub[2])
    
    img = save_image(fname=fname)
    
    return img, np.mean(success_rates)
    
    
    
def plot_generator_samples(generator, env=None, size=100, fname=None):
    
    goals, _ = generator.sample_states(size)
    
    if env is None:
        limit = np.max(np.abs(goals))
        goal_lb = -np.array([limit, limit, limit])
        goal_ub = np.array([limit, limit, limit])
    else:
        goal_lb = env.goal_lb
        goal_ub = env.goal_ub
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], s=10)
    ax.set_xlim(goal_lb[0], goal_ub[0])
    ax.set_ylim(goal_lb[1], goal_ub[1])
    ax.set_zlim(goal_lb[2], goal_ub[2])
    
    img = save_image(fname=fname)
    return img

        
        
    
    
    
    