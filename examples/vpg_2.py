from __future__ import print_function
from rllab.mdp.box2d.cartpole_mdp import CartpoleMDP
from rllab.policy.mean_std_nn_policy import MeanStdNNPolicy
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.mdp.normalized_mdp import normalize
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam

# normalize() makes sure that the actions for the MDP lies within the range
# [-1, 1]
mdp = normalize(CartpoleMDP())
# Initialize a neural network policy with a single hidden layer of 32 hidden
# units
policy = MeanStdNNPolicy(mdp, hidden_sizes=[32])
# Initialize a linear baseline estimator using state features
baseline = LinearFeatureBaseline(mdp)

# We will collect 100 trajectories per iteration
N = 100
# Each trajectory will have at most 100 time steps
T = 100
# Number of iterations
n_itr = 100
# Set the discount factor for the problem
discount = 0.99
# Learning rate for the gradient update
learning_rate = 0.1

# Construct the computation graph

observations_var = TT.matrix('observations')
actions_var = TT.matrix('actions')
advantages_var = TT.vector('advantages')

# policy.get_log_prob_sym computes the symbolic log probability of the
# actions given the observations
# Note that we negate the objective, since most optimizers assume a
# minimization problem
surr = - TT.mean(policy.get_log_prob_sym(observations_var,
                                         actions_var) * advantages_var)
# Get the list of trainable parameters
params = policy.get_params(trainable=True)
grads = theano.grad(surr, params)

f_train = theano.function(
    inputs=[observations_var, actions_var, advantages_var],
    outputs=None,
    updates=adam(grads, params, learning_rate=learning_rate),
    allow_input_downcast=True
)

for _ in xrange(n_itr):

    paths = []

    for _ in xrange(N):
        observations = []
        actions = []
        rewards = []

        observation = mdp.reset()

        for _ in xrange(T):
            # policy.get_action() returns a pair of values. The second one
            # summarizes the distribution of the actions in the case of a
            # stochastic policy. This information is useful when forming
            # importance sampling ratios. In our case it is not needed.
            action, _ = policy.get_action(observation)
            next_observation, reward, terminal = mdp.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            observation = next_observation
            if terminal:
                break

        # We need to compute the empirical return for each time step along the
        # trajectory
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
        )

        path_baseline = baseline.predict(path)
        advantages = []
        return_so_far = 0
        for t in xrange(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            advantage = return_so_far - path_baseline[t]
            advantages.append(advantage)
        # The advantages are stored backwards in time, so we need to revert it
        advantages = np.array(advantages[::-1])

        advantages = (advantages - np.mean(advantages)) / \
            (np.std(advantages) + 1e-8)

        path["advantages"] = advantages

        paths.append(path)

    observations = np.concatenate([p["observations"] for p in paths])
    actions = np.concatenate([p["actions"] for p in paths])
    advantages = np.concatenate([p["advantages"] for p in paths])

    f_train(observations, actions, advantages)
    print('Average Return:', np.mean([sum(p["rewards"]) for p in paths]))
