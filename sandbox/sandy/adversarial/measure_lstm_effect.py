#!/usr/bin/env python

from collections import defaultdict
from chainer import functions as F
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import seaborn as sns
from pandas import DataFrame

from sandbox.sandy.adversarial.plot_seaborn import NORMS
from sandbox.sandy.misc.util import create_dir_if_needed
from sandbox.sandy.shared.model import load_models
from sandbox.sandy.shared.model_rollout import get_average_return_a3c

SHOW_PLOT = True
EXPERIMENTS = ['async-rl_exp038', 'async-rl_exp037']
GAMES = ['chopper', 'pong', 'seaquest', 'space']
SAVED_MODEL_DIR = '/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/trained_models_recurrent'
OUTPUT_DIR = '/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/lstm_effect_plots'

EXP_TO_DESC = dict(exp037="4", exp038="1")

def compute_lstm_grad(nn, out, state, saved_grads):
    # This should be called by async_rl.agents.a3c_agent right after the input
    # is passed all the way through the model
    # Each entry of saved_grads is
    # (max_prev_c_grad, mean_prev_c_grad, max_prev_h_grad,
    #  mean_prev_h_grad, max_state_grad, mean_state_grad)
    # where max and mean are taken across all the entries of nn.lstm_prev_c.grad
    # and nn.lstm_prev_h.grad

    logits = nn.pi.compute_logits(out)
    max_logits = F.broadcast_to(F.max(logits), (1,len(logits)))

    # Loss between predicted action distribution and the action distribution that
    # places all weight on the argmax action
    ce_loss = F.log(1.0 / F.sum(F.exp(logits - max_logits)))

    # Clear gradients, just in case
    for x in [out, logits, max_logits, ce_loss]:
        x.cleargrad()
    nn.cleargrads()

    ce_loss.backward(retain_grad=True)
    grad_prev_c = nn.lstm_prev_c.grad
    grad_prev_h = nn.lstm_prev_h.grad
    grad_state = state.grad

    # Clear gradients again
    for x in [out, logits, max_logits, ce_loss]:
        x.cleargrad()
    nn.cleargrads()
    
    saved_grads.append((np.max(abs(grad_prev_c)), np.mean(abs(grad_prev_c)), \
                        np.max(abs(grad_prev_h)), np.mean(abs(grad_prev_h)),
                        np.max(abs(grad_state)), np.mean(abs(grad_state))))

def compute_lstm_grads(game, exp, base_dir, threshold=0.8, num_threshold=3, \
                       seed=1, deterministic=True):
    # Load policies that satisfy threshold and num_threshold criteria
    policies = load_models([game], [exp], base_dir, threshold=threshold, \
                           num_threshold=num_threshold)
    policies = list(itertools.chain.from_iterable(policies[game].values()))
    print([x.score for x in policies])

    # For each policy, run single rollout. At each point during the rollout,
    # compute gradient of output action with respect to LSTM cell state.
    all_policy_grads = []
    for policy in policies:
        assert type(policy.algo.cur_agent.model).__name__ == "A3CLSTM"
        policy.env.set_adversary_fn(None)
        all_grads = []
        fn = lambda x, y, z: compute_lstm_grad(x, y, z, all_grads)
        policy.algo.cur_agent.model.set_callback_fn(fn)
        avg_return, paths, timesteps = get_average_return_a3c(policy.algo, \
                seed, N=1, deterministic=deterministic)
        all_policy_grads.append(all_grads)
    return all_policy_grads

def plot_grads(game, exp, grads, output_dir, norm='l-inf'):
    # Plot mean and std. dev. of gradients, specify that these are magnitudes
    grads_df_dict = defaultdict(list)
    gradient_names = ['max, '+r'$\nabla_c$', 'mean, '+r'$\nabla_c$', \
                      'max, '+r'$\nabla_h$', 'mean, '+r'$\nabla_h$', \
                      'max, '+r'$\nabla_x$', 'mean, '+r'$\nabla_x$']
    gradient_order = ['mean, '+r'$\nabla_c$', 'mean, '+r'$\nabla_h$', \
                      'mean, '+r'$\nabla_x$', 'max, '+r'$\nabla_c$', \
                      'max, '+r'$\nabla_h$', 'max, '+r'$\nabla_x$']
    for exp, exp_grads in grads:
        for policy_grads in exp_grads:
            for grad in policy_grads:
                for i in range(len(grad)):
                    if gradient_names[i] == 'max, '+r'$\nabla_x$':
                        continue
                    grads_df_dict['Number of Frames'].append(EXP_TO_DESC[exp.split('_')[1]])
                    grads_df_dict['Gradient'].append(gradient_names[i])
                    grads_df_dict['value'].append(grad[i])

    grads_df = DataFrame(grads_df_dict)
    sns.set_palette(sns.color_palette('muted'))
    ax = sns.barplot(x="Number of Frames", y="value", hue="Gradient", data=grads_df, \
                     hue_order=gradient_order)
    ax.set(ylabel='magnitude')

    if game == "chopper":
        game = "chopper-command"
    elif game == "space":
        game = "space-invaders"
    game_cap = ' '.join([word.capitalize() for word in game.split('-')])
    fig_title = game_cap + ", " + NORMS[norm] + ' norm'
    sns.plt.title(fig_title)
    if SHOW_PLOT:
        sns.plt.show()
    extension = '.png'
    ax.get_figure().savefig(osp.join(output_dir, game + '_nomaxx' + extension), \
                            bbox_inches='tight', pad_inches=0.0)
    sns.plt.clf()

def main():
    create_dir_if_needed(OUTPUT_DIR)
    for game in GAMES:
        all_grads = []
        for exp in EXPERIMENTS:
            grads = compute_lstm_grads(game, exp, SAVED_MODEL_DIR)
            all_grads.append((exp, grads))
        plot_grads(game, exp, all_grads, OUTPUT_DIR)

if __name__ == "__main__":
    main()
