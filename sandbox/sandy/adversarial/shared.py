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

def set_seed(algo, seed):
    # Set random seed for policy rollouts
    #ext.set_seed(seed)
    parallel_sampler.set_seed(seed)

    # Set random seed of Atari environment
    if hasattr(algo.env, 'ale'):  # envs/atari_env_haoran.py
        algo.env.set_seed(seed)
    elif hasattr(algo.env, '_wrapped_env'):  # Means algo.env is a ProxyEnv
        base_env = get_base_env(algo.env._wrapped_env)
        base_env._seed(seed)
    elif hasattr(algo.env, 'env'):  # envs/atari_env.py
        base_env = get_base_env(algo.env)
        base_env._seed(seed)
    else:
        raise Exception("Invalid environment")

def get_average_return(algo, seed=None):
    # Note that batch size is set during load_model
    if seed is not None:  # Set random seed, for reproducibility
        set_seed(algo, seed)

    paths = algo.sampler.obtain_samples(None)
    avg_return = np.mean([sum(p['rewards']) for p in paths])
    return avg_return, paths

def load_model(params_file, batch_size):
    # Load model from saved file (e.g., params.pkl or itr_##.pkl from rllab)
    data = joblib.load(params_file)
    algo = data['algo']
    algo.batch_size = batch_size
    algo.sampler.worker_batch_size = batch_size
    algo.n_parallel = 1
    try:
        algo.max_path_length = data['env'].horizon
    except NotImplementedError:
        algo.max_path_length = 50000

    # Copying what happens at the start of algo.train() -- to initialize workers
    assert type(algo).__name__ in ['TRPO', 'ParallelTRPO'], "Algo type not supported"
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
