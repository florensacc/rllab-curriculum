
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.distributions.categorical import Categorical
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc import ext
import numpy as np
import theano
import theano.tensor as TT
import multiprocessing as mp
from lasagne.updates import adam
from contextlib import closing


def new_shared_mem_array(init_val):
    typecode = init_val.dtype.char
    arr = mp.RawArray(typecode, np.prod(init_val.shape))
    nparr = np.frombuffer(arr)#.get_obj())
    nparr.shape = init_val.shape
    return nparr


def init_worker(x):
    global x_
    x_ = x
    x_var = theano.shared(x_, borrow=True)
    global f_update
    f_update = ext.compile_function(
        inputs=[],
        outputs=[],
        updates=[(x_var, x_var + TT.ones_like(x_var))]
    )


def update_x(*args, **kwargs):
    global f_update
    f_update()


def main():
    x = np.zeros(10)  # random.normal(size=(env.observation_space.flat_dim, 32)) * 0.1
    x = new_shared_mem_array(x)
    with closing(mp.Pool(initializer=init_worker, initargs=(x,))) as p:
        p.map(update_x, range(10000))
        #p.map(update_x, [0, 1, 2])
        #p.map(update_x, [0, 1, 2])
    print(x)

main()

# env = GridWorldEnv(desc='4x4_safe')
# # Manually construct a neural network with a single hidden layer
#
# W1 = theano.shared(W1_val, name="W")
# b1_val = np.zeros((1, 32))
# b1 = theano.shared(b1_val, name="b", broadcastable=(True, False))
#
# W2_val = np.random.normal(size=(32, env.action_space.flat_dim)) * 0.1
# W2 = theano.shared(W2_val, name="W")
# b2_val = np.zeros((1, env.action_space.flat_dim))
# b2 = theano.shared(b2_val, name="b", broadcastable=(True, False))
#
#
# def prob_sym(obs_var):
#     h = TT.nnet.relu(obs_var.dot(W1) + b1)
#     return TT.nnet.softmax(h.dot(W2) + b2)
#
#
# obs_var = TT.matrix('obs')
#
# f_prob = ext.compile_function(inputs=[obs_var], outputs=prob_sym(obs_var))
#
#
# def dist_info_sym(obs_var, action_var):
#     return dict(prob=prob_sym(obs_var))
#
#
# def get_params(**tags):
#     return [W1, b1, W2, b2]
#
#
# def get_action(observation):
#     prob = f_prob([env.observation_space.flatten(observation)])[0]
#     action = env.action_space.weighted_sample(prob)
#     return action, dict(prob=prob)
#
#
# policy = ext.AttrDict(
#     dist_info_sym=dist_info_sym,
#     distribution=Categorical(),
#     get_params=get_params,
#     get_action=get_action,
# )
#
# # policy = GaussianMLPPolicy(env.spec, hidden_sizes=(32,))
# # Initialize a linear baseline estimator using state features
# baseline = LinearFeatureBaseline(env.spec)
#
# # We will collect 100 trajectories per iteration
# N = 100
# # Each trajectory will have at most 100 time steps
# T = 100
# # Number of iterations
# n_itr = 100
# # Set the discount factor for the problem
# discount = 0.99
# # Learning rate for the gradient update
# learning_rate = 0.01
#
# # Construct the computation graph
#
# observations_var = TT.matrix('observations')
# actions_var = TT.imatrix('actions')
# advantages_var = TT.vector('advantages')
#
# # policy.get_log_prob_sym computes the symbolic log probability of the
# # actions given the observations
# # Note that we negate the objective, since most optimizers assume a
# # minimization problem
# dist_info_vars = policy.dist_info_sym(observations_var, actions_var)
# dist = policy.distribution
# surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)
# # Get the list of trainable parameters
# params = policy.get_params(trainable=True)
# grads = theano.grad(surr, params)
#
# f_train = theano.function(
#     inputs=[observations_var, actions_var, advantages_var],
#     outputs=None,
#     updates=adam(grads, params, learning_rate=learning_rate),
#     allow_input_downcast=True
# )
#
# for _ in xrange(n_itr):
#
#     paths = []
#
#     for _ in xrange(N):
#         observations = []
#         actions = []
#         rewards = []
#
#         observation = env.reset()
#
#         for _ in xrange(T):
#             # policy.get_action() returns a pair of values. The second one
#             # summarizes the distribution of the actions in the case of a
#             # stochastic policy. This information is useful when forming
#             # importance sampling ratios. In our case it is not needed.
#             action, _ = policy.get_action(observation)
#             next_observation, reward, terminal, _ = env.step(action)
#             observations.append(env.observation_space.flatten(observation))
#             actions.append(env.action_space.flatten(action))
#             rewards.append(reward)
#             observation = next_observation
#             if terminal:
#                 break
#
#         # We need to compute the empirical return for each time step along the
#         # trajectory
#         path = dict(
#             observations=np.array(observations),
#             actions=np.array(actions),
#             rewards=np.array(rewards),
#         )
#
#         path_baseline = baseline.predict(path)
#         advantages = []
#         return_so_far = 0
#         for t in xrange(len(rewards) - 1, -1, -1):
#             return_so_far = rewards[t] + discount * return_so_far
#             advantage = return_so_far - path_baseline[t]
#             advantages.append(advantage)
#         # The advantages are stored backwards in time, so we need to revert it
#         advantages = np.array(advantages[::-1])
#
#         advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
#
#         path["advantages"] = advantages
#
#         paths.append(path)
#
#     observations = np.concatenate([p["observations"] for p in paths])
#     actions = np.concatenate([p["actions"] for p in paths])
#     advantages = np.concatenate([p["advantages"] for p in paths])
#
#     f_train(observations, actions, advantages)
#     print('Average Return:', np.mean([sum(p["rewards"]) for p in paths]))
