import copy
import joblib
import numpy as np
from rllab.misc import ext
from rllab.sampler import parallel_sampler

def get_base_env(obj):
    # Find level of obj that contains base environment, i.e., the env that links to ALE
    # (New version of Monitor in OpenAI gym adds an extra level of wrapping)
    while True:
        if not hasattr(obj, 'env'):
            return None
        if hasattr(obj.env, 'ale'):
            return obj.env
        else:
            obj = obj.env

def set_seed_env(env, seed):
    # Set random seed for policy rollouts
    #ext.set_seed(seed)
    parallel_sampler.set_seed(seed)

    # Set random seed of Atari environment
    if hasattr(env, 'ale'):  # envs/atari_env_haoran.py
        env.set_seed(seed)
    elif hasattr(env, '_wrapped_env'):  # Means env is a ProxyEnv
        base_env = get_base_env(env._wrapped_env)
        base_env._seed(seed)
    elif hasattr(env, 'env'):  # envs/atari_env.py
        base_env = get_base_env(env)
        base_env._seed(seed)
    else:
        raise Exception("Invalid environment")

def set_seed(algo, seed):
    set_seed_env(algo.env, seed)

def get_average_return_trpo(algo, seed, N=10):
    # Note that batch size is set during load_model

    # Set random seed, for reproducibility
    set_seed(algo, seed)
    curr_seed = seed + 1

    #paths = algo.sampler.obtain_samples(None)
    paths = []
    while len(paths) < N:
        new_paths = algo.sampler.obtain_samples(n_samples=1)  # Returns single path
        paths.append(new_paths[0])
        set_seed(algo, curr_seed)
        curr_seed += 1

    avg_return = np.mean([sum(p['rewards']) for p in paths])
    return avg_return, paths

def get_average_return_a3c(algo, seed, N=10, horizon=10000):
    algo.test_env = copy.deepcopy(algo.cur_env)
    # copy.deepcopy doesn't copy lambda function
    algo.test_env.adversary_fn = algo.cur_env.adversary_fn
    algo.test_agent = copy.deepcopy(algo.cur_agent)

    # Set random seed, for reproducibility
    set_seed_env(algo.test_env, seed)
    curr_seed = seed + 1

    #scores, paths = algo.evaluate_performance(N, horizon, return_paths=True)
    paths = []
    while len(paths) < N:
        _, new_paths = algo.evaluate_performance(1, horizon, return_paths=True)
        paths.append(new_paths[0])

        algo.test_env = copy.deepcopy(algo.cur_env)
        # copy.deepcopy doesn't copy lambda function
        algo.test_env.adversary_fn = algo.cur_env.adversary_fn
        algo.test_agent = copy.deepcopy(algo.cur_agent)
        set_seed_env(algo.test_env, curr_seed)
        curr_seed += 1

    avg_return = np.mean([sum(p['rewards']) for p in paths])
    return avg_return, paths

def get_average_return(algo, seed, N=10):
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        return get_average_return_trpo(algo, seed, N=N)
    elif algo_name in ['A3CALE']:
        return get_average_return_a3c(algo, seed, N=N)
    else:
        assert False, "Algorithm type " + algo_name + " is not supported."

def load_model_trpo(algo):
    algo.batch_size = batch_size
    algo.sampler.worker_batch_size = batch_size
    algo.n_parallel = 1
    try:
        algo.max_path_length = data['env'].horizon
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

def load_model_a3c(algo):
    return algo, algo.cur_env

def load_model(params_file, batch_size):
    # Load model from saved file (e.g., params.pkl or itr_##.pkl from rllab)
    data = joblib.load(params_file)
    algo = data['algo']
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        return load_model_trpo(algo)
    elif algo_name in ['A3CALE']:
        return load_model_a3c(algo)
    else:
        assert False, "Algorithm type " + algo_name + " is not supported."
