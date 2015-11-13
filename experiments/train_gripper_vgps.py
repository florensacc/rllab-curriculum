
# coding: utf-8

# # Variational Guided Policy Search
# For each iteration:
# 
# - Run iLQG to find time-varying Gaussian controller
# - Run lbfgs to fit the policy

# In[1]:

import os
os.environ['TENSORFUSE_MODE'] = 'cgt'
from mdp.gripper_mdp import GripperMDP
from algo.optim import ilqg
from misc.ext import extract
import numpy as np


# Compute marginals
# 
# $$ \mu_{t+1} = \begin{bmatrix}f_{xt} & f_{ut} \end{bmatrix} \begin{bmatrix} \mu_t \cr \bar u_t + k_t + K_t(\mu_t - \bar x_t) \end{bmatrix}$$
# $$ \Sigma_{t+1} = \begin{bmatrix} f_{xt} & f_{ut} \end{bmatrix} \begin{bmatrix} \Sigma_t & \Sigma_t K_t^T \\ K_t \Sigma_t & Q_{uut}^{-1} + K_t \Sigma_t K_t^T \end{bmatrix}\begin{bmatrix} f_{xt} & f_{ut} \end{bmatrix} ^T + \Sigma_{ft} $$

# In[2]:

# Compute marginals
# how to choose the initial covariance?
#Dx = x0.shape[0]

def compute_marginals(mdp, mu0, Sigma0, Sigmaf, xref, uref, K, k, Quu):
    Dx = mu0.shape[0]
    from algo.optim.ilqg import linearize
    fx, fu = extract(
        linearize(xref, uref, mdp.forward_dynamics, mdp.cost, mdp.final_cost),
        "fx", "fu"
    )
    T = len(uref)
    mu = np.zeros((T, Dx))
    Sigma = np.zeros((T, Dx, Dx))
    log_std = np.zeros((T, Dx))
    mu[0] = mu0
    Sigma[0] = Sigma0
    for t in range(T-1):
        mu[t+1] = fx[t].dot(mu[t] - xref[t]) + fu[t].dot(K[t].dot(mu[t] - xref[t])) + xref[t+1]
        Sigma[t+1] = np.bmat([fx[t], fu[t]]).dot(
            np.bmat([
                [Sigma[t], Sigma[t].dot(K[t].T)],
                [K[t].dot(Sigma[t]), np.linalg.inv(Quu[t]) + K[t].dot(Sigma[t]).dot(K[t].T)]
            ])).dot(np.bmat([fx[t], fu[t]]).T) + Sigmaf
    return mu, Sigma

#print k
# print 'mu[30]:', qx_mu[30], 'shape:', qx_mu.shape
# print 'xref[30]:', xref[30], 'shape:', xref.shape
# print 'Sigma[30]:', qx_Sigma[30], 'shape:', qx_Sigma.shape


# Optimize $\mathcal{L}(q, \theta)$ with respect to $\theta$. We represent the policy as a deterministic policy with additive Gaussian noise. We have
# 
# $$ \mathcal{L}(q, \theta) = \int q(\xi) \log \frac{p(\mathcal{O} | \xi) p(\xi | \theta)}{q(\xi)} d\xi$$
# 
# We can sample states from the marginals $q(x_t)$, and then evaluate the integral over the actions analytically, obtaining
# 
# $$ \mathcal{L}(q, \theta) \approx \frac{1}{M} \sum_{i=1}^M \sum_{t=1}^T \int q(u_t|x_t) \log \pi_\theta (u_t | x_t) du_t + const$$
# 
# Let $q(u_t | x_t^i) = \mathcal{G}(u_t; \mu_{x_t^i}^q, \Sigma_{x_t^i}^q)$, $\pi_\theta(u_t | x_t^i) = \mathcal{G}(u_t; \mu_{x_t^i}^\theta, \Sigma_{x_t^i}^\theta)$. We have
# 
# $$ \mathcal{L}(q, \theta) \approx \frac{1}{M}\sum_{i=1}^M \sum_{t=1}^T -\frac{1}{2} (\mu_{x_t^i}^\theta - \mu_{x_t^i}^q) ^T {\Sigma_{x_t^i}^\theta}^{-1} (\mu_{x_t^i}^\theta - \mu_{x_t^i}^q) - \frac{1}{2} \log |\Sigma_{x_t^i}^\theta| - \frac{1}{2} tr({\Sigma_{x_t^i}^\theta}^{-1} \Sigma_{x_t^i}^q) + const$$
# 

# In[10]:

from policy.mujoco_policy import MujocoPolicy
import tensorfuse as theano
import tensorfuse.tensor as T
from misc.tensor_utils import flatten_tensors
from scipy.optimize import fmin_l_bfgs_b
from policy.linear_gaussian_policy import LinearGaussianPolicy


def train_policy(nn_policy, trajopt_policy, qx_mu, qx_Sigma, n_samples=10):
    q_mean = T.matrix("q_mean")
    q_log_std = T.matrix("q_log_std")
    q_pdist = T.concatenate([q_mean, q_log_std], axis=1)
    # this is what we're going to minimize
    loss = T.mean(policy.kl(q_pdist, policy.pdist_var))
    input_var = policy.input_var

    grads = theano.grad(loss, policy.params)

    f_loss = theano.function([input_var, q_mean, q_log_std], loss, allow_input_downcast=True, on_unused_input='ignore')
    f_grads = theano.function([input_var, q_mean, q_log_std], grads, allow_input_downcast=True, on_unused_input='ignore')

    qxs = []
    qu_means = []
    qu_log_stds = []
    for i in range(n_samples):
        for t in range(len(qx_mu)):
            xt = np.random.multivariate_normal(qx_mu[t], qx_Sigma[t])
            mean, log_std = trajopt_policy.get_pdist(xt, t)
            qxs.append(xt)
            qu_means.append(mean)
            qu_log_stds.append(log_std)

    qxs = np.vstack(qxs)
    qu_means = np.vstack(qu_means)
    qu_log_stds = np.vstack(qu_log_stds)

    def evaluate_cost(params):
        policy.set_param_values(params)
        return f_loss(qxs, qu_means, qu_log_stds)

    def evaluate_grads(params):
        policy.set_param_values(params)
        return flatten_tensors(f_grads(qxs, qu_means, qu_log_stds)).astype(np.float64)

    print 'loss before:', f_loss(qxs, qu_means, qu_log_stds)
    result = fmin_l_bfgs_b(func=evaluate_cost, x0=policy.get_param_values(), fprime=evaluate_grads, maxiter=20)
    print 'loss after:', f_loss(qxs, qu_means, qu_log_stds)

# initial setup
mdp = GripperMDP()
x0, _ = mdp.reset()
Dx = len(x0)
uinit = np.zeros((mdp.horizon, mdp.action_dim))
Sigma0 = np.eye(Dx) * 0.01
Sigmaf = np.eye(Dx) * 0.01
policy = MujocoPolicy(mdp, hidden_sizes=[10])
alpha_max = 1.0
alpha_min = 0.1
alpha_decay_range = 50
n_iter = 100

# Initialize trajectory
print "obtaining initial trajectory"
ilqg_result = ilqg.solve(x0, uinit, sysdyn=mdp.forward_dynamics, cost_func=mdp.cost, final_cost_func=mdp.final_cost)
xref, uref, K, k, Quu = extract(ilqg_result, "x", "u", "K", "k", "Quu")


for itr in range(n_iter):
    print "iteration #%d" % itr
    # Compute marginals
    qx_mu, qx_Sigma = compute_marginals(mdp, x0, Sigma0, Sigmaf, xref, uref, K, k, Quu)
    # Optimize L(q, theta)
    print "training policy"
    train_policy(policy, LinearGaussianPolicy(xref=xref, uref=uref, K=K, k=k, Quu=Quu), qx_mu, qx_Sigma)
    # Compute alpha
    if itr >= alpha_decay_range:
        alpha = alpha_min
    else:
        alpha = np.exp(
            (alpha_decay_range - itr - 1) * 1.0 / alpha_decay_range * np.log(alpha_max) + \
            (itr + 1) * 1.0 / alpha_decay_range * np.log(alpha_min)
        )
    print alpha
    # Re-optimize trajectory
    print "re-optimize trajectory"
    def scaled_cost(x, u):
        return alpha*mdp.cost(x, u) - policy.get_action_log_prob(x, u)
    ilqg_result = ilqg.solve(x0, uinit, sysdyn=mdp.forward_dynamics, cost_func=scaled_cost, final_cost_func=mdp.final_cost)
    xref, uref, K, k, Quu = extract(ilqg_result, "x", "u", "K", "k", "Quu")
    import sys
    sys.stdout.flush()
