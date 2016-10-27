from rllab.misc.nb_utils import ExperimentDatabase
import matplotlib.pyplot as plt
import os
import joblib
import json
import numpy as np
import itertools


# counting modes from the unpickled data (policy)
def cluster(policy, all_latent_dists_at_0, free_config, modes, min_dist_modes, k):  # k is the configuration index
    my_dist = all_latent_dists_at_0[k]
    my_mode = modes[k]
    for i in free_config:
        if policy.distribution.kl(my_dist, all_latent_dists_at_0[i]) < min_dist_modes:
            free_config.remove(
                i)  # maybe better using remove (the first element, but in this case they are all different)
            modes[i] = my_mode
            cluster(policy, all_latent_dists_at_0, free_config, modes, min_dist_modes, i)
    return


def count_modes(data_unpickle):
    policy = data_unpickle['policy']
    env = data_unpickle['env']

    all_latents = [np.array(i) for i in itertools.product([0, 1], repeat=policy.latent_dim)]
    all_latent_dists_at_0 = []
    for lat in all_latents:
        policy.set_pre_fix_latent(lat)
        print("pre-setting the latent to: ", lat)
        all_latent_dists_at_0.append(policy.get_action(np.array((0, 0)))[1])
        print("the path had latent: ", all_latent_dists_at_0[-1]['latents'])
    policy.unset_pre_fix_latent()

    # count the number of good modes: at a distance of 0.1 of any of the n good modes:
    good_modes_affiliation = np.zeros(env.n)
    mu = env.mu
    A = np.array([[np.cos(2. * np.pi / env.n), -np.sin(2. * np.pi / env.n)],
                  [np.sin(2. * np.pi / env.n), np.cos(2. * np.pi / env.n)]])  # rotation matrix
    good_modes_means = [np.dot(np.linalg.matrix_power(A, k), mu) for k in range(env.n)]
    print("all good modes are: ", good_modes_means)
    distance_between_good_modes = np.linalg.norm(mu - good_modes_means[1], 2)
    min_dist_modes = distance_between_good_modes / 5.
    for i, info_dist in enumerate(all_latent_dists_at_0):
        print('latent {} produces mean {}'.format(info_dist['latents'], info_dist['mean']))
        for j, good_mode_mean in enumerate(good_modes_means):
            if np.linalg.norm(info_dist['mean'] - good_mode_mean, 2) < min_dist_modes:
                print('latent {} is affiliated to good_mode {}\n'.format(info_dist['latents'], j))
                good_modes_affiliation[j] += 1
    print(good_modes_affiliation)
    num_good_modes = np.count_nonzero(good_modes_affiliation)

    free_config = list(range(len(all_latents)))
    modes = [0] * len(free_config)
    mode_num = 0
    while free_config:
        mode_num += 1
        k = free_config[0]
        free_config.remove(k)
        modes[k] = mode_num
        cluster(policy, all_latent_dists_at_0, free_config, modes, min_dist_modes, k)
    print(modes)
    return mode_num, num_good_modes


## plot for all the experiments
def analyze_modes(datadir):
    analyzed_modes = []  # let's have a list of tuples, the first elem the 3params and the second the #modes
    latent_dims = []  # let's also save the parameters encountered.
    rew_coefs = []
    n_hallus = []
    database = ExperimentDatabase(datadir, names_or_patterns='*')
    # check if you're giving the final dir of an experiment

    exps = database._experiments
    if not len(exps):
        database = ExperimentDatabase(datadir, )

    for i, exp in enumerate(exps):
        # get the last pickle
        exp_name = exp.params['exp_name']
        if os.path.isdir(os.path.join(datadir, exp_name)):
            path_experiment = os.path.join(datadir, exp_name)
            print("Analyzing the exp in: ", path_experiment)
        else:
            path_experiment = datadir
        pkl_file = 'params.pkl'
        json_name = 'params.json'
        json_file = os.path.join(path_experiment, json_name)
        with open(json_file) as data_file:
            all_params = json.load(data_file)
        last_data_unpickle = joblib.load(os.path.join(path_experiment, pkl_file))

        policy = last_data_unpickle['policy']

        num_modes, num_good_modes = count_modes(last_data_unpickle)
        last_true_rew = exp.progress['TrueAverageReturn'][-1]
        rew_coef = all_params['json_args']['algo']['reward_coef_mi']
        n_hallu = all_params['json_args']['algo']['hallucinator']['n_hallucinate_samples']
        latent_dim = policy.latent_dim
        if rew_coef not in rew_coefs:
            rew_coefs.append(rew_coef)
        if n_hallu not in n_hallus:
            n_hallus.append(n_hallu)
        if latent_dim not in latent_dims:
            latent_dims.append(latent_dim)
        analyzed_modes.append(([latent_dim, rew_coef, n_hallu], num_modes, num_good_modes, last_true_rew))
    return analyzed_modes, np.sort(latent_dims), np.sort(rew_coefs), np.sort(n_hallus)


def plot_modes(analyzed_modes, latent_dims, rew_coefs, n_hallus):
    print('plotting modes')
    matrix_modes = np.zeros((len(latent_dims), len(rew_coefs), len(n_hallus)))
    matrix_good_modes = np.zeros((len(latent_dims), len(rew_coefs), len(n_hallus)))
    matrix_last_true_rew = np.zeros((len(latent_dims), len(rew_coefs), len(n_hallus)))

    matrix_modes_std = np.zeros((len(latent_dims), len(rew_coefs), len(n_hallus)))
    matrix_good_modes_std = np.zeros((len(latent_dims), len(rew_coefs), len(n_hallus)))
    matrix_last_true_rew_std = np.zeros((len(latent_dims), len(rew_coefs), len(n_hallus)))

    sorted_rew_coefs = np.sort(rew_coefs)  # just in case, but they should already be sorted
    sorted_n_hallus = np.sort(n_hallus)
    sorted_latent_dims = np.sort(latent_dims)
    print(sorted_latent_dims, sorted_rew_coefs, sorted_n_hallus)
    i = 0
    while i < len(analyzed_modes):
        mode = analyzed_modes[i]
        sum_modes = [mode[1]]
        sum_good_modes = [mode[2]]
        sum_last_true_rews = [mode[3]]
        n = 1
        i += 1
        while i < len(analyzed_modes) and mode[0] == analyzed_modes[i][0]:
            sum_modes.append(analyzed_modes[i][1])
            sum_good_modes.append(analyzed_modes[i][2])
            sum_last_true_rews.append(analyzed_modes[i][3])
            i += 1
            n += 1
        idx_latent_dim = np.where(sorted_latent_dims == mode[0][0])
        idx_rew_coef = np.where(sorted_rew_coefs == mode[0][1])
        idx_n_hallu = np.where(sorted_n_hallus == mode[0][2])
        matrix_modes[idx_latent_dim, idx_rew_coef, idx_n_hallu] = np.mean(sum_modes)
        matrix_good_modes[idx_latent_dim, idx_rew_coef, idx_n_hallu] = np.mean(sum_good_modes)
        matrix_last_true_rew[idx_latent_dim, idx_rew_coef, idx_n_hallu] = np.mean(sum_last_true_rews)

        matrix_modes_std[idx_latent_dim, idx_rew_coef, idx_n_hallu] = np.std(sum_modes)
        matrix_good_modes_std[idx_latent_dim, idx_rew_coef, idx_n_hallu] = np.std(sum_good_modes)
        matrix_last_true_rew_std[idx_latent_dim, idx_rew_coef, idx_n_hallu] = np.std(sum_last_true_rews)
    return matrix_modes, matrix_good_modes, matrix_last_true_rew, \
           matrix_modes_std, matrix_good_modes_std, matrix_last_true_rew_std


## plot for all the experiments
if __name__ == "__main__":
    import sys

    # name_dir=sys.argv[1]
    # path_dir = "./data/local/"+name_dir
    path_dir = sys.argv[1]
    print("counting modes in all experiments in: " + path_dir)
    analyzed_modes, latent_dims, rew_coefs, n_hallus = analyze_modes(path_dir)
    params_experiments = dict(latent_dims=latent_dims, rew_coefs=rew_coefs, n_hallus=n_hallus)
    matrix_modes, matrix_good_modes, matrix_last_true_rew, \
        matrix_modes_std, matrix_good_modes_std, matrix_last_true_rew_std= plot_modes(analyzed_modes, latent_dims, rew_coefs, n_hallus)
    save_file = os.path.join(path_dir, 'mode_matrices.pkl')
    save_dic = dict(params_experiments=params_experiments,
                    matrix_modes=matrix_modes,
                    matrix_good_modes=matrix_good_modes,
                    matrix_last_true_rew=matrix_last_true_rew,
                    matrix_modes_std=matrix_modes_std,
                    matrix_good_modes_std=matrix_good_modes_std,
                    matrix_last_true_rew_std=matrix_last_true_rew_std,
                    )
    joblib.dump(save_dic, save_file)
    print('modes: \n', matrix_modes, '\ngood modes\n', matrix_good_modes, '\nlast True Rew:\n', matrix_last_true_rew)
