from rllab.envs.proxy_env import ProxyEnv
def _worker_compute_jacobians(G,s,a,fd_step):
    def forward(state,action):
        """
        Beware: not resetting to the original env state
        """
        G.env.reset(state)
        next_observation,reward,done,env_info = G.env.step(action)
        return next_observation, reward

    s2,r = forward(s,a)
    s_dim = len(s)
    a_dim = len(a)

    f_s = np.zeros((s_dim,s_dim))
    r_s = np.zeros((1,s_dim))
    for i in range(s_dim):
        ds = np.zeros_like(s)
        ds[i] = fd_step
        new_s = s + ds
        new_s2, new_r = forward(new_s,a)
        f_s[:,i] = (new_s2 - s2) / fd_step
        r_s[:,i] = (new_r - r) / fd_step

    f_a = np.zeros((s_dim,a_dim))
    r_a = np.zeros((1,a_dim))
    for i in range(a_dim):
        da = np.zeros_like(a)
        da[i] = fd_step
        new_a = a + da
        new_s2, new_r = forward(s,new_a)
        f_a[:,i] = (new_s2 - s2) / fd_step
        r_a[:,i] = (new_r - r) / fd_step

    return f_s,f_a,r_s,r_a

from rllab.sampler import parallel_sampler
def compute_approximate_value_gradient(env,policy,path,gamma,fd_step):
    args_list = [
        [s,a,fd_step]
        for s,a in zip(path["observations"],path["actions"])
    ]
    results = parallel_sampler.singleton_pool.run_map(
        _worker_compute_jacobians, args_list
    )
    path["f_s"] = []
    path["f_a"] = []
    path["r_s"] = []
    path["r_a"] = []
    for result in results:
        path["f_s"].append(result[0])
        path["f_a"].append(result[1])
        path["r_s"].append(result[2])
        path["r_a"].append(result[3])

    return compute_analytic_value_gradient(env,policy,path,gamma)


def compute_analytic_value_gradient(env,policy,path,gamma):
    precomputed_env_jacobians = "r_a" in path.keys()

    T = len(path["rewards"]) # path length
    states = path["observations"]
    n = len(states[0]) # state dimension
    D = len(policy.get_theta()) # parameter dimension
    V_theta = np.zeros((T,D)) # gradient wrt params at all time steps
    V_s = np.zeros((T,n)) # gradient wrt states at all time steps

    for t in range(T-2,-1,-1):
        s = states[t]
        a = path["actions"][t]
        s2 = states[t+1]
        if precomputed_env_jacobians:
            r_a = path["r_a"][t]
            r_s = path["r_s"][t]
            f_s = path["f_s"][t]
            f_a = path["f_a"][t]
        else:
            r_a = env.r_a(s,a)
            r_s = env.r_s(s,a)
            f_s = env.f_s(s,a)
            f_a = env.f_a(s,a)
        pi_theta = policy.pi_theta(s)
        pi_s = policy.pi_s(s)

        V_s[t] = r_s + r_a.dot(pi_s) + gamma * V_s[t+1].dot(f_s + f_a.dot(pi_s))
        V_theta[t] = r_a.dot(pi_theta) + gamma * V_s[t+1].dot(f_a.dot(pi_theta)) + gamma * V_theta[t+1]
    return V_s,V_theta


def central_finite_difference_jacobian(func,inputs,output_dim,step,wrt=0):
    n = len(inputs[wrt])
    m = output_dim
    J = np.zeros((m,n))
    def f(wrt_input):
        _inputs = inputs
        _inputs[wrt] = wrt_input
        return func(*_inputs)

    wrt_input = inputs[wrt]
    for i in range(n):
        d_wrt_input = np.zeros(n)
        d_wrt_input[i] = step
        output_plus = f(wrt_input + d_wrt_input)
        output_minus = f(wrt_input - d_wrt_input)
        J[:,i] = (output_plus - output_minus).ravel() / (2. * step)
    return J


import numpy as np
from rllab.misc import tensor_utils
import time

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,record_full_state=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    if record_full_state:
        s = env._full_state
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        if record_full_state:
            env_info["full_states"] = s
            s = env._full_state
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
