#!/usr/bin/env python

import argparse
import h5py
import os.path as osp
import seaborn as sns
from pandas import DataFrame

from sandbox.sandy.adversarial.io_util import get_param_names, get_all_result_paths

PLOT_TYPE = ["no-transfer"]  # Options: ['no-transfer', 'transfer-policy', 'transfer-algo']
PLOT_TARGET = ['exp027']
PLOT_ADV = ['exp027']
SAVE_AS_PDF = False

def plot_returns(h5_file, cond_key, x_key, y_key, exp_to_algo, screen_print=False):
    # cond_key, x_key, y_key are 2-tuples (key_in_h5_file, key_for_plot)
    # Structure of h5_file should be:
    #     y = f[group][--stuff--][y_key[0]][()]

    param_names = get_param_names(h5_file)
    policy_adv_idx = param_names.index('policy_adv')
    policy_target_idx = param_names.index('policy_rollout')

    f = h5py.File(h5_file, 'r')
    data = []
    paths = get_all_result_paths(h5_file, y_key[0])
    paths_by_game_exp = {}  # keys: (game, exp_index)
    for path in paths:
        path_info = path.split('/')[2:]
        policy_adv_info = path_info[policy_adv_idx].split('_')
        policy_target_info = path_info[policy_target_idx].split('_')
        exp_idx  = policy_target_info[0]
        game = policy_target_info[-1]
        if game == 'invaders':
            game = 'space-invaders'
        elif game == 'command':
            game = 'chopper-command'
        if game not in paths_by_game_exp:
            paths_by_game_exp[game] = {}
        if exp_idx not in paths_by_game_exp[game]:
            paths_by_game_exp[game][exp_idx] = []
        include = False
        if "no-transfer" in PLOT_TYPE and path_info[policy_adv_idx] == path_info[policy_target_idx]:
            include = True
        elif "transfer-policy" in PLOT_TYPE and policy_adv_info[0] == policy_target_info[0]:
            include = True
        elif "transfer-algo" in PLOT_TYPE and policy_adv_info[0] != policy_target_info[0]:
            include = True
        if include and policy_adv_info[0] in PLOT_ADV and policy_target_info[0] in PLOT_TARGET:
            paths_by_game_exp[game][exp_idx].append((path, f[path][y_key[0]][()]))
    f.close()

    games = sorted(paths_by_game_exp.keys())
    for game_idx, game in enumerate(games):
        for exp_idx in paths_by_game_exp[game]:
            # Generate plot

            data = [path.split('/')[2:] + [y] \
                    for path, y in paths_by_game_exp[game][exp_idx]]
            df_dict = {}
            for i, p_name in enumerate(param_names):
                p_name_plot = p_name
                if p_name_plot == x_key[0]:
                    p_name_plot = x_key[1]
                elif p_name_plot == cond_key[0]:
                    p_name_plot = cond_key[1]

                if p_name == x_key[0]:
                    df_dict[p_name_plot] = [float(x[i]) for x in data]
                else:
                    df_dict[p_name_plot] = [x[i] for x in data]
            df_dict[y_key[1]] = [x[-1] for x in data]
            cond_x_count = {}
            cond_idx = param_names.index(cond_key[0])
            x_idx = param_names.index(x_key[0])
            df_dict['subject'] = []
            for d in data:
                cond_x = (d[cond_idx],d[x_idx])
                if cond_x not in cond_x_count:
                    cond_x_count[cond_x] = 0
                df_dict['subject'].append(cond_x_count[cond_x])
                cond_x_count[cond_x] += 1
                
            # Create DataFrame
            df = DataFrame(df_dict)

            sns.set(style="darkgrid")
            # Plot the response with standard error
            ax = sns.tsplot(data=df, time=x_key[1], unit="subject", \
                           condition=cond_key[1], value=y_key[1], ci=68, \
                           interpolate=True, legend=True)
            #ax.set(xlim=(0, 0.018)) #, ylim=(-.05, 1.05))
            #ax.set(xlabel=x_key, ylabel=y_key)
            #plt.subplots_adjust(top=0.9)
            game_cap = ' '.join([word.capitalize() for word in game.split('-')])
            fig_title = game_cap + ", " + exp_to_algo[exp_idx]
            sns.plt.suptitle(fig_title)

            print("Showing plot")
            sns.plt.show()
            #sns.plt.savefig(osp.join(osp.dirname(h5_file), game + '_' + exp_to_algo[exp_idx] + '.pdf'))
            if SAVE_AS_PDF:
                extension = '.pdf'
            else:
                extension = '.png'
            ax.get_figure().savefig(osp.join(osp.dirname(h5_file), game + '_' + exp_to_algo[exp_idx] + extension))

            sns.plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    plot_returns(args.returns_h5, ('norm', 'Norm'), ('fgsm_eps', r'$\epsilon$'), ('avg_return', 'Average Return'), \
                 {'exp027':"TRPO", 'exp036':'A3C'}, screen_print=True)

if __name__ == "__main__":
    main()
