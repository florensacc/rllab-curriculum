#!/usr/bin/env python

import itertools
import numpy as np
import os.path as osp

from sandbox.sandy.adversarial.fgsm_sleeper import fgsm_sleeper_perturbation
from sandbox.sandy.adversarial.io_util import save_performance_to_all, \
        init_output_file, init_all_output_file
from sandbox.sandy.misc.util import get_time_stamp, create_dir_if_needed
from sandbox.sandy.shared.experiment import get_experiments
from sandbox.sandy.shared.model import TrainedModel, load_models
from sandbox.sandy.shared.model_rollout import get_average_return

BASE_DIR = "/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/output_sleeper_Feb20"
#MODEL_DIR = "exp037_20170208_230001_954345_chopper_command"
MODEL_DIR = "exp038_20170213_155825_910994_chopper_command"
BASE_MODEL_DIR = "/home/shhuang/src/rllab-private/data/s3"
SAVED_MODEL_DIR = '/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/trained_models_recurrent'
TEST_TRANSFER = False

def sleeper_adv(policy_adv, policy_target, norm, fgsm_eps, seed, \
                k=1, ts=None, deterministic=True, \
                adversary_algo='fgsm', obs_min=0, obs_max=1, \
                save_rollouts=False, num_ts=100):
    # k - number of time steps of delay for adversarial perturbation; i.e.,
    #     perturbation happens at time t, target policy acts normally for times
    #     t through t+k-1, does different action at time t+k
    # ts - time steps for adversary to make perturbation at; can either be
    #     integers (corresponding to actual time steps), or floats between 0 and
    #     1 (fraction of the total trajectory length). If None, num_ts timesteps
    #     are taken at even intervals along the rollout
    # deterministic - if True, target policy always picks argmax action to
    #     execute> If False, policy sampled from output distribution over actions.

    # Do complete rollout to figure out its length
    policy_target.env.set_adversary_fn(None)
    policy_target.algo.cur_agent.model.lstm.reset_state()
    avg_return_noadv, paths, timesteps = \
            get_average_return(policy_target.algo, seed, N=1, \
                               return_timesteps=True, deterministic=deterministic)
    print(avg_return_noadv, timesteps)

    # Sample ts if not given; convert ts to timesteps if necessary
    if ts is None:
        ts = np.linspace(0,1,num_ts,endpoint=False)
    orig_ts = np.array(ts)
    ts = np.array(ts)
    if len([x for x in ts if x > 0 and x < 1]) > 0:
        assert (ts >= 0).all() and (ts <= 1).all()
        ts = ts * timesteps
    ts = np.round(ts).astype(int)

    output_dir = osp.join(BASE_DIR, get_time_stamp())
    create_dir_if_needed(output_dir)

    output_h5 = None
    if save_rollouts:
        output_fname = "{algo_name}_{norm}_{eps}_{policy_adv}_{policy_target}.h5".format(
            algo_name=policy_adv.exp.algo_name,
            norm=norm,
            eps=str(fgsm_eps).replace('.', '-'),
            policy_adv=policy_adv.model_name,
            policy_target=policy_target.model_name
        )
        output_h5 = init_output_file(output_dir, \
                                     None, 'fgsm', \
                                     {'eps': fgsm_eps, 'norm': norm, 'ts': ts, \
                                     'orig_ts': orig_ts, 'k': k}, \
                                     fname=output_fname, \
                                     algo_name=policy_adv.exp.algo_name)

    all_output_h5 = init_all_output_file(output_dir, adversary_algo,
                                         ['norm', 'fgsm_eps', 'policy_adv', \
                                          'policy_rollout', 'k', 't'])
    print("all_output_h5:", all_output_h5)
    print("output_h5:", output_h5)

    # For each t, run policy rollout (without any adversarial perturbations)
    # until that time step, then perturb at time t and branch into 
    # two rollouts (for current and next k timesteps, i.e., time steps t through
    # t+k) after perturbing at time t vs. rollout without perturbing
    # at time t; save the two sequences of output action distributions
    # (to output_h5) to see if indeed only the last one (at time step t+k) is changed
    assert type(policy_adv.algo).__name__ == "A3CALE"
    policy_adv.algo.cur_agent.model.skip_unchain = True
    #all_paths = []
    
    for t in ts:
        policy_target.algo.cur_agent.model.lstm.reset_state()
        policy_adv.algo.cur_agent.model.lstm.reset_state()

        if adversary_algo == 'fgsm':
            adv_params = dict(
                    t=t,
                    k=k,
                    fgsm_eps=fgsm_eps,
                    norm=norm,
                    obs_min=obs_min,
                    obs_max=obs_max,
                    output_h5=output_h5,
                    policy_adv=policy_adv.model_name,
                    policy_rollout=policy_target.model_name
            )
            adversary_fn = lambda x,y: fgsm_sleeper_perturbation(x, y, policy_adv.algo, **adv_params)
        else:
            raise NotImplementedError
        policy_target.env.set_adversary_fn(adversary_fn)

        avg_return_adv, paths = \
                get_average_return(policy_target.algo, seed, N=1, \
                                   return_timesteps=False, deterministic=deterministic)
        #all_paths.append((t, avg_return_adv, paths))
        save_performance_to_all(all_output_h5, avg_return_adv, adv_params, len(paths))

def main():
    experiments = ['async-rl_exp038', 'async-rl_exp037']
    games = ['chopper', 'pong', 'seaquest', 'space']
    norms = ['l-inf']
    fgsm_eps = [0.0005, 0.001, 0.002, 0.004]
    ks = [1,2,3,4,5]
    seed = 1

    for game in games:
        for exp in experiments:
            for n in norms:
                for k in ks:
                    if game == 'chopper' and exp == 'async-rl_exp038' and k < 4:
                        print("Skipping previously calculated: chopper, exp038, k =", k)
                        continue
                    for e in fgsm_eps:
                        print(game, exp, n, e, k)
                        adv_policies = load_models([game], [exp], SAVED_MODEL_DIR, 0.80, 1)
                        target_policies = load_models([game], [exp], SAVED_MODEL_DIR, 0.80, 1)
                        adv_game_policies = list(itertools.chain.from_iterable(adv_policies[game].values()))
                        target_game_policies = list(itertools.chain.from_iterable(target_policies[game].values()))
                        for policy_adv in adv_game_policies:
                            for policy_target in target_game_policies:
                                if not TEST_TRANSFER and \
                                        policy_adv.model_name != policy_adv.model_name:
                                    continue
                                sleeper_adv(policy_adv, policy_target, n, e, seed, k=k, ts=None, \
                                            deterministic=True, save_rollouts=True, num_ts=100)

    #experiment = get_experiments(['chopper'], ['async-rl_exp037'], BASE_MODEL_DIR)[0]
    #experiment = get_experiments(['chopper'], ['async-rl_exp038'], BASE_MODEL_DIR)[0]
    #policy_adv = TrainedModel(MODEL_DIR, experiment)
    #policy_target = TrainedModel(MODEL_DIR, experiment)
    #norm = 'l-inf'
    #fgsm_eps = 0.004
    #seed = 1

    #ks = [1,2,3,4,5]
    #for k in ks:
    #    sleeper_adv(policy_adv, policy_target, norm, fgsm_eps, seed, k=k, ts=None, \
    #                deterministic=True, save_rollouts=True, num_ts=100)

if __name__ == "__main__":
    main()
