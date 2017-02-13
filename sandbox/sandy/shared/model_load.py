#!/usr/bin/env python

import joblib

BATCH_SIZE = 100000

class DQNAlgo(object):
    pass

def load_model_dqn(data):
    from sandbox.sandy.deep_q_rl import ale_data_set
    algo = DQNAlgo()
    algo.agent = data['agent']
    # Initialize algo.agent.test_data_set (which isn't pickled), which is
    # used to keep history of the last n_frames inputs
    algo.agent.test_data_set = ale_data_set.DataSet(
            width=algo.agent.image_width,
            height=algo.agent.image_height,
            max_steps=algo.agent.phi_length * 2,
            phi_length=algo.agent.phi_length)
    algo.agent.testing = True
    algo.agent.bonus_evaluator = None
    algo.env = data['env']
    return algo, algo.env

def load_model_trpo(data, batch_size=BATCH_SIZE):
    algo = data['algo']
    algo.batch_size = batch_size
    algo.sampler.worker_batch_size = batch_size
    algo.n_parallel = 1
    try:
        algo.max_path_length = algo.env.horizon
    except NotImplementedError:
        algo.max_path_length = 50000

    # Copying what happens at the start of algo.train() -- to initialize workers
    if 'TRPO' == type(algo).__name__:
        algo.start_worker()
    algo.init_opt()
    if 'ParallelTRPO' in str(type(algo)):
        algo.init_par_objs()

    if hasattr(algo.env, "_wrapped_env"):  # Means algo.env is a ProxyEnv
        env = algo.env._wrapped_env
    else:
        env = algo.env
    return algo, env

def load_model(model_file):
    # Model is saved differently for each deep RL algorithm
    data = joblib.load(model_file)
    if 'algo' in data:
        algo_name = type(data['algo']).__name__
        if algo_name in ['TRPO', 'ParallelTRPO']:  # TRPO
            return load_model_trpo(data)
        elif algo_name in ['A3CALE']:  # A3C
            return data['algo'], data['algo'].cur_env
    elif 'agent' in data:  # DQN
        return load_model_dqn(data)
    raise NotImplementedError
