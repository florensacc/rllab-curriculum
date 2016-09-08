import joblib
import numpy as np
import sys
from matplotlib import pyplot as plt


def plot_series(series):
    plt.figure(1)
    # colors = [np.array([1, 0.1, 0.1]), np.array([0.1, 1, 0.1]), np.array([0.1, 0.1, 1])]
    colors = ['m', 'g', 'r', 'b', 'y']
    for i, s in enumerate(series):
        print(s['x'], s['y'], s['std'], s['label'])
        small_number = np.ones_like(s['x']) * (s['x'][1]*0.1)
        x_axis = np.where(s['x'] == 0, small_number, s['x'])
        plt.plot(x_axis, s['y'], color=colors[i], label=s['label'])
        plt.fill_between(x_axis, s['y'] - s['std'], s['y'] + s['std'], color=colors[i], alpha=0.2)
    plt.semilogx()
    plt.xlabel('MI reward bonus')
    plt.ylabel('Final intrinsic reward')
    plt.title('Final intrinsic reward in pointMDP with 10 good modes')
    plt.legend(loc='best')
    plt.show()


def give_coord_and_label(params_to_plot, all_params):
    coord_and_labels = []
    x_axis = []
    latent_dim_to_plot = params_to_plot['latent_dims']
    rew_coef_to_plot = params_to_plot['rew_coefs']
    n_hallu_to_plot = params_to_plot['n_hallus']
    for lat in latent_dim_to_plot:
        if lat == -1:
            x_axis = all_params['latent_dims']
            lat_coord = list(range(len(x_axis)))
            lat_label = ''
        else:
            lat_coord = np.where(all_params['latent_dims'] == lat)
            lat_label = 'latent_dim: ' + str(lat)
        for coef in rew_coef_to_plot:
            if coef == -1:
                x_axis = all_params['rew_coefs']
                coef_coord = list(range(len(x_axis)))
                coef_label = ''
            else:
                coef_coord = np.where(all_params['rew_coefs'] == coef)
                coef_label = ', coef_rew: ' + str(coef)
            for hallu in n_hallu_to_plot:
                if hallu == -1:
                    x_axis = all_params['n_hallus']
                    hallu_coord = list(range(len(x_axis)))
                    hallu_label = ''
                else:
                    hallu_coord = np.where(all_params['n_hallus'] == hallu)
                    hallu_label = ', n_hallu: ' + str(hallu)
                coord_and_labels.append(([lat_coord, coef_coord, hallu_coord],
                                         lat_label + coef_label + hallu_label))
    return coord_and_labels, x_axis


def give_series(list_of_coords_and_labels, x_axis, matrix, matrix_std):
    series = []
    for coord_label in list_of_coords_and_labels:
        series.append(dict(x=np.array(x_axis),
                           y=np.array(matrix[coord_label[0]]).reshape((-1,)),
                           std=np.array(matrix_std[coord_label[0]]).reshape((-1,)),
                           label=coord_label[1])
                      )
    return series


if __name__ == '__main__':
    pkl_file = sys.argv[1]
    saved_dict = joblib.load(pkl_file)
    params_dict = saved_dict['params_experiments']
    matrix_good_modes = saved_dict['matrix_good_modes']
    matrix_last_true_reward = saved_dict['matrix_last_true_rew']
    matrix_modes = saved_dict['matrix_modes']

    matrix_good_modes_std = saved_dict['matrix_good_modes_std']
    matrix_last_true_reward_std = saved_dict['matrix_last_true_rew_std']
    matrix_modes_std = saved_dict['matrix_modes_std']

    params_to_plot = dict(latent_dims=[3],
                          rew_coefs=[-1],
                          n_hallus=[0,1,2,4])

    coord_and_labels, x_axis = give_coord_and_label(params_to_plot, params_dict)
    series = give_series(coord_and_labels, x_axis, matrix_last_true_reward, matrix_last_true_reward_std)
    plot_series(series)
