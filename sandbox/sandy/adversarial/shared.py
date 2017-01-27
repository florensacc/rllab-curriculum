import copy
import joblib
import numpy as np
import os, os.path as osp

SCORE_KEY = {'async-rl': 'ReturnAverage', \
             'deep-q-rl': 'TestAverageReturn', \
             'trpo': 'RawReturnAverage'}

class DQNAlgo(object):
    pass

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
    from rllab.sampler import parallel_sampler
    #from rllab.misc import ext
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
    timesteps = sum([len(p['rewards']) for p in paths])
    return avg_return, paths, timesteps

def get_average_return_a3c(algo, seed, N=10, horizon=10000):

    #scores, paths = algo.evaluate_performance(N, horizon, return_paths=True)
    paths = []
    curr_seed = seed
    while len(paths) < N:
        algo.test_env = copy.deepcopy(algo.cur_env)
        # copy.deepcopy doesn't copy lambda function
        algo.test_env.adversary_fn = algo.cur_env.adversary_fn
        algo.test_agent = copy.deepcopy(algo.cur_agent)
        # Set random seed, for reproducibility
        set_seed_env(algo.test_env, curr_seed)

        _, new_paths = algo.evaluate_performance(1, horizon, return_paths=True)
        paths.append(new_paths[0])

        del algo.test_env
        del algo.test_agent
        curr_seed += 1

    avg_return = np.mean([sum(p['rewards']) for p in paths])
    timesteps = sum([len(p['rewards']) for p in paths])
    return avg_return, paths, timesteps

def sample_dqn(algo, n_paths=1):  # Based on deep_q_rl/ale_experiment.py, run_episode
    env = algo.env
    #paths = [{'rewards':[], 'states':[], 'actions':[]} for i in range(n_paths)]
    rewards = [0]*n_paths
    timesteps = 0
    for i in range(n_paths):
        env.reset()
        action = algo.agent.start_episode(env.last_state)
        total_reward = 0
        while True:
            if env.is_terminal:
                algo.agent.end_episode(env.reward)
                break
            #paths[i]['states'].append(env.observation)
            #paths[i]['actions'].append(action)
            env.step(action)
            #paths[i]['rewards'].append(env.reward)
            total_reward += env.reward
            action = algo.agent.step(env.reward, env.last_state, {})
            timesteps += 1
        rewards[i] = total_reward
    return rewards, timesteps

def get_average_return_dqn(algo, seed, N=10):
    # Set random seed, for reproducibility
    set_seed(algo, seed)
    curr_seed = seed + 1

    rewards = [0]*N
    total_timesteps = 0
    for i in range(N):
        new_rewards, timesteps = sample_dqn(algo, n_paths=1)  # Returns single path
        rewards[i] = new_rewards[0]
        total_timesteps += timesteps
        set_seed(algo, curr_seed)
        curr_seed += 1

    avg_return = np.mean(rewards)
    return avg_return, rewards, total_timesteps
    
def get_average_return(algo, seed, N=10, return_timesteps=False):
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        get_average_return_f = get_average_return_trpo
    elif algo_name in ['A3CALE']:
        get_average_return_f = get_average_return_a3c
    elif algo_name in ['DQNAlgo']:
        get_average_return_f = get_average_return_dqn
    else:
        assert False, "Algorithm type " + algo_name + " is not supported."
    avg_return, paths, timesteps = get_average_return_f(algo, seed, N=N)
    if return_timesteps:
        return avg_return, paths, timesteps
    else:
        return avg_return, paths

def load_model_trpo(algo, env, batch_size):
    algo.batch_size = batch_size
    algo.sampler.worker_batch_size = batch_size
    algo.n_parallel = 1
    try:
        algo.max_path_length = env.horizon
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

def load_model(params_file, batch_size):
    # Load model from saved file (e.g., params.pkl or itr_##.pkl from rllab)
    data = joblib.load(params_file)

    if 'algo' in data:
        algo = data['algo']
        algo_name = type(algo).__name__
        if algo_name in ['TRPO', 'ParallelTRPO']:  # TRPO
            return load_model_trpo(algo, data['env'], batch_size)
        elif algo_name in ['A3CALE']:  # A3C
            return algo, algo.cur_env
        else:
            assert False, "Algorithm type " + algo_name + " is not supported."
    elif 'agent' in data:  # DQN
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
    else:
        raise NotImplementedError

def load_models(games, experiments, base_dir, batch_size, threshold=0, \
                num_threshold=5, score_window=10):
    # each entry in experiments should have the format "algo-name_exp-index"
    # If threshold is in [0,1], then discard all policies which have a score
    # less than threshold * the best policy's score - for each game and
    # training algorithm pair

    policies = {}  # key, top level: game name
                   # key, second level: algorithm name
                   # value: list of (algo, env) pairs - trained policies for that game
    for game in games:
        policies[game] = {}
        for exp in experiments:
            algo_name, exp_index = exp.split('_')
            if algo_name not in policies[game]:
                policies[game][algo_name] = []

            params_parent_dir = osp.join(base_dir, algo_name, exp_index+'-'+game)
            params_dirs = [osp.join(params_parent_dir, x) for x in os.listdir(params_parent_dir)]
            params_dirs = [x for x in params_dirs if osp.isdir(x)]
            for params_dir in params_dirs:
                params_files = [x for x in os.listdir(params_dir) \
                                if x.startswith('itr') and x.endswith('pkl')]
                # Get the latest parameters
                params_file = sorted(params_files,
                                     key=lambda x: int(x.split('.')[0].split('_')[1]),
                                     reverse=True)[0]
                itr = int(params_file.split('.')[0].split('_')[1])
                params_file = osp.join(params_dir, params_file)
                algo, env = load_model(params_file, batch_size)
                
                # Calculate average score starting from saved iteration and
                # going back score_window iterations
                with open(osp.join(params_dir, 'progress.csv'), 'r') as progress_f:
                    lines = progress_f.readlines()
                    header = lines[0].split(',')
                    score_idx = header.index(SCORE_KEY[algo_name])
                    scores = [float(l.split(',')[score_idx]) \
                              for l in lines[max(1,itr-score_window):itr+1]]
                    score = sum(scores) / float(len(scores))
                policies[game][algo_name].append((algo, env, score, params_file.split('/')[-2]))

    if threshold > 1:
        threshold = 1

    # Discard all policies that are not close to as good as the best one, or not
    # in the top num_threshold scores
    best_policies = {}
    for game in policies:
        best_policies[game] = {}

        for algo_name in policies[game]:
            all_policies = sorted(policies[game][algo_name], key=lambda x: x[2], reverse=True)
            best_score = all_policies[0][2]
            best_policies[game][algo_name] = [x for x in all_policies if x[2] >= best_score*threshold]
            best_policies[game][algo_name] = best_policies[game][algo_name][:num_threshold]

    return best_policies
