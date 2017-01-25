#!/usr/bin/env python

import argparse
import h5py
import os.path as osp
import seaborn as sns
from pandas import DataFrame

from sandbox.sandy.adversarial.io_util import get_param_names, get_all_result_paths

def plot_returns(h5_file, cond_key, x_key, y_key, exp_to_algo, screen_print=False):
    # Structure of h5_file should be:
    #     y = f[group][--stuff--][y_key][()]

    f = h5py.File(h5_file, 'r')
    data = []
    paths = get_all_result_paths(h5_file, y_key)
    paths_by_game_exp = {}  # keys: (game, exp_index)
    for path in paths:
        path_info = (path.split('/')[-1]).split('_')  # Assumes exp_name is last param
        game = path_info[-1]
        if game == 'invaders':
            game = 'space-invaders'
        elif game == 'command':
            game = 'chopper-command'
        exp_idx = path_info[0]
        if game not in paths_by_game_exp:
            paths_by_game_exp[game] = {}
        if exp_idx not in paths_by_game_exp[game]:
            paths_by_game_exp[game][exp_idx] = []
        paths_by_game_exp[game][exp_idx].append((path, f[path][y_key][()]))
    f.close()

    param_names = get_param_names(h5_file)

    for game in paths_by_game_exp:
        for exp_idx in paths_by_game_exp[game]:
            # Generate plot

            data = [path.split('/')[2:] + [y] \
                    for path, y in paths_by_game_exp[game][exp_idx]]
            df_dict = {}
            for i, p_name in enumerate(param_names):
                if p_name == x_key:
                    df_dict[p_name] = [float(x[i]) for x in data]
                else:
                    df_dict[p_name] = [x[i] for x in data]
            df_dict[y_key] = [x[-1] for x in data]
            cond_x_count = {}
            cond_idx = param_names.index(cond_key)
            x_idx = param_names.index(x_key)
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
            ax = sns.tsplot(data=df, time=x_key, unit="subject", \
                           condition=cond_key, value=y_key, ci=68, \
                           interpolate=False, legend=True)
            #ax.set(xlim=(0, 0.018)) #, ylim=(-.05, 1.05))
            #ax.set(xlabel=x_key, ylabel=y_key)
            #plt.subplots_adjust(top=0.9)
            game_cap = ' '.join([word.capitalize() for word in game.split('-')])
            fig_title = game_cap + ", " + exp_to_algo[exp_idx]
            sns.plt.suptitle(fig_title)

            print("Showing plot")
            sns.plt.show()
            osp.dirname(h5_file)
            sns.plt.savefig(osp.join(osp.dirname(h5_file), game + '_' + exp_to_algo[exp_idx] + '.pdf'))

            sns.plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    plot_returns(args.returns_h5, 'norm', 'fgsm_eps', 'avg_return', \
                 {'exp027':"TRPO", 'exp036':'A3C'}, screen_print=True)

if __name__ == "__main__":
    main()
