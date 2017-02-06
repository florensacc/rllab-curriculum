#!/usr/bin/env python

import argparse
import h5py
import os.path as osp
import seaborn as sns
from pandas import DataFrame

from sandbox.sandy.adversarial.io_util import get_param_names, get_all_result_paths

# Options: ['no-transfer', 'transfer-policy', 'transfer-algorithm']
#PLOT_TYPES = ['no-transfer', 'transfer-policy', 'transfer-algorithm']
PLOT_TYPES = ['no-transfer']
PLOT_TARGET = ['exp027', 'exp035c', 'exp036']
PLOT_ADV = ['exp027', 'exp035c', 'exp036']
SAVE_AS_PDF = True
SHOW_PLOT = False
NORMS = {'l1': r'$\ell 1$',
         'l2': r'$\ell 2$',
         'l-inf': r'$\ell \infty$'}
ORDER = {'Algorithm': 1,
         'Policy': 2,
         'None': 3}

def plot_returns(h5_file, cond_key, x_key, y_key, exp_to_algo, screen_print=False):
    # cond_key, x_key, y_key are 2-tuples (key_in_h5_file, key_for_plot)
    # Structure of h5_file should be:
    #     y = f[group][--stuff--][y_key[0]][()]

    param_names = get_param_names(h5_file)
    policy_adv_idx = param_names.index('policy_adv')
    policy_target_idx = param_names.index('policy_rollout')

    paths = get_all_result_paths(h5_file, y_key[0])

    f = h5py.File(h5_file, 'r')
    data = []
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
        path_type = ""
        if "no-transfer" in PLOT_TYPES and path_info[policy_adv_idx] == path_info[policy_target_idx]:
            include = True
            path_type = "None"
        elif "transfer-policy" in PLOT_TYPES and policy_adv_info[0] == policy_target_info[0]:
            include = True
            path_type = "Policy"
        elif "transfer-algorithm" in PLOT_TYPES and policy_adv_info[0] != policy_target_info[0]:
            include = True
            path_type = "Algorithm"
        if include and policy_adv_info[0] in PLOT_ADV and policy_target_info[0] in PLOT_TARGET:
            paths_by_game_exp[game][exp_idx].append((path, f[path][y_key[0]][()], path_type))
    f.close()

    games = sorted(paths_by_game_exp.keys())
    for game_idx, game in enumerate(games):
        for exp_idx in paths_by_game_exp[game]:
            # Generate plot

            data = [path.split('/')[2:] + [y, path_type] \
                    for path, y, path_type in paths_by_game_exp[game][exp_idx]]
            sns.set(style="darkgrid")

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
                    params = [x[i] for x in data]
                    params = [NORMS[p] if p in NORMS else p for p in params]
                    df_dict[p_name_plot] = params

            df_dict[y_key[1]] = [x[-2] for x in data]
            cond_x_count = {}
            cond_idx = param_names.index(cond_key[0])
            x_idx = param_names.index(x_key[0])
            df_dict['subject'] = []
            df_dict['Norm and Transfer Type'] = []
            df_dict['Transfer Type'] = []
            df_dict['Transfer Type Order'] = []
            for d in data:
                cond_x = (d[cond_idx],d[x_idx],d[-1])
                if cond_x not in cond_x_count:
                    cond_x_count[cond_x] = 0
                df_dict['subject'].append(cond_x_count[cond_x])
                if d[cond_idx] in NORMS:
                    cond = NORMS[d[cond_idx]]
                else:
                    cond = d[cond_idx]
                df_dict['Norm and Transfer Type'].append(cond + ", " + d[-1].capitalize())
                df_dict['Transfer Type'].append(d[-1].capitalize())
                df_dict['Transfer Type Order'].append(ORDER[d[-1]])
                cond_x_count[cond_x] += 1
                
            # Create DataFrame
            df = DataFrame(df_dict)
            df.sort(['Transfer Type Order'], ascending=[1])

            # Plot the response with standard error
            if len(PLOT_TYPES) == 1:
                sns.set_context("paper", font_scale=2)
                sns.plt.figure(figsize=(8, 6))
                ax = sns.tsplot(data=df, time=x_key[1], unit="subject", \
                               condition=cond_key[1], value=y_key[1], ci=68, \
                               interpolate=True, legend=False,
                               color={NORMS['l-inf']: '#8172b2',
                                      NORMS['l2']: '#ccb974',
                                      NORMS['l1']: '#64b5cd'})
                game_cap = ' '.join([word.capitalize() for word in game.split('-')])
                fig_title = game_cap + ", " + exp_to_algo[exp_idx]
                sns.plt.title(fig_title)
                if SHOW_PLOT:
                    sns.plt.show()
                if SAVE_AS_PDF:
                    extension = '.pdf'
                else:
                    extension = '.png'
                ax.get_figure().savefig(osp.join(osp.dirname(h5_file), game + '_' + exp_to_algo[exp_idx] + extension), bbox_inches='tight', pad_inches=0.0)

                sns.plt.clf()

            else:
                sns.set_context("paper", font_scale=1.8)
                sns.plt.figure(figsize=(8, 6))
                # Plot norms separately
                for norm_key in NORMS.keys():
                    norm = NORMS[norm_key]
                    ax = sns.tsplot(data=df.loc[df['FGSM Norm'] == norm], \
                                    time=x_key[1], unit="subject", \
                                    condition="Transfer Type", value=y_key[1], ci=68, \
                                    interpolate=True, legend=False,
                                    color={"Algorithm": '#c44e52', "None": '#55a868', "Policy": '#4c72b0'}) #err_style="ci_bars")
                    # Algorithm = red, Policy = blue, None = green
                    #ax.legend(['Algorithm', 'Policy', 'None'], title="Transfer Type")
                    #ax.add_legend(label_order = ['Algorithm', 'Policy', 'None'])
                    game_cap = ' '.join([word.capitalize() for word in game.split('-')])
                    fig_title = game_cap + ", " + exp_to_algo[exp_idx] + ", " + norm + " norm"
                    sns.plt.title(fig_title)

                    if SHOW_PLOT:
                        sns.plt.show()
                    if SAVE_AS_PDF:
                        extension = '.pdf'
                    else:
                        extension = '.png'
                    ax.get_figure().savefig(osp.join(osp.dirname(h5_file), game + '_' + exp_to_algo[exp_idx] + '_' + norm_key + extension), bbox_inches='tight', pad_inches=0.0)
                    sns.plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    plot_returns(args.returns_h5, ('norm', 'FGSM Norm'), ('fgsm_eps', r'$\epsilon$'), ('avg_return', 'Average Return'), \
            {'exp027':"TRPO", 'exp035c':'DQN', 'exp036':'A3C'}, screen_print=True)

if __name__ == "__main__":
    main()
