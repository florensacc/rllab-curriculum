"""AdvExperiment class runs rollouts of a policy with adversarial perturbations
"""

import itertools, os.path as osp

from rllab.misc import logger

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation
from sandbox.sandy.adversarial.io_util import init_output_file, \
        save_performance, save_video_file, init_all_output_file, save_performance_to_all
from sandbox.sandy.adversarial.shared import get_average_return, load_model, load_models, get_latest_param_file
from sandbox.sandy.adversarial.vis import visualize_adversary
from sandbox.sandy.misc.util import get_time_stamp, to_iterable

EXP_IDX_TO_NAME = {'exp027': 'trpo_exp027',
                   'exp036': 'async-rl_exp036',
                   'exp035c': 'deep-q-rl_exp035c'}

class AdvExperiment(object):
    def __init__(self, games, norms, fgsm_eps, experiments, model_dir, \
                 save_rollouts, test_transfer, adversary_algo, \
                 adversary_algo_param_names, seed, obs_min=0, obs_max=1, \
                 batch_size=20000, threshold_perf=0.8, threshold_n=3, N=10,
                 video_params={}, policy_adv=None, policy_target=None):
        self.games = to_iterable(games)
        self.norms = to_iterable(norms)
        self.fgsm_eps = to_iterable(fgsm_eps)
        self.experiments = to_iterable(experiments)
        self.policy_adv = policy_adv
        self.policy_target = policy_target

        if self.games is None or self.experiments is None:
            assert self.policy_adv is not None and self.policy_target is not None
        self.model_dir = model_dir
        self.save_rollouts = save_rollouts
        self.test_transfer = test_transfer
        self.adversary_algo = adversary_algo
        self.adversary_algo_param_names = adversary_algo_param_names
        self.seed = seed
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.batch_size = batch_size
        self.threshold_perf = threshold_perf
        self.threshold_n = threshold_n
        self.N = N
        self.video_params=video_params

    def run_for_adv_target(self, game, policy_adv, policy_target, norm, fgsm_eps, algo_name, all_output_h5):
        logger.record_tabular("Game", game)
        logger.record_tabular("AdversaryPolicy", policy_adv[3])
        logger.record_tabular("TargetPolicy", policy_target[3])
        logger.record_tabular("Norm", norm)
        logger.record_tabular("FGSMEps", fgsm_eps)
        output_h5 = None
        if self.save_rollouts:
            output_fname = "{algo_name}_{norm}_{eps}_{policy_adv}_{policy_target}.h5".format(
                algo_name=algo_name,
                norm=norm,
                eps=str(fgsm_eps).replace('.', '-'),
                policy_adv=policy_adv[3],
                policy_target=policy_target[3]
            )
            output_h5 = init_output_file(logger.get_snapshot_dir(), \
                                         None, 'fgsm', \
                                         {'eps': fgsm_eps, 'norm': norm}, \
                                         fname=output_fname, \
                                         algo_name=algo_name)

        if self.adversary_algo == 'fgsm':
            adv_params = dict(
                    fgsm_eps=fgsm_eps,
                    norm=norm,
                    obs_min=self.obs_min,
                    obs_max=self.obs_max,
                    output_h5=output_h5,
                    policy_adv=policy_adv[3],
                    policy_rollout=policy_target[3]
            )
            adversary_fn = lambda x: fgsm_perturbation(x, policy_adv[0], **adv_params)
        else:
            raise NotImplementedError

        # Run policy rollouts with FGSM adversary for N trials, get average return
        policy_target[1].set_adversary_fn(adversary_fn)
        avg_return_adversary, paths, timesteps = \
                get_average_return(policy_target[0], self.seed, \
                                   N=self.N, return_timesteps=True)
        logger.record_tabular('Timesteps', timesteps)
        path_lengths = [len(p['rewards']) for p in paths]
        if self.save_rollouts:
            save_performance(output_h5, avg_return_adversary, len(paths), path_lengths)

            output_prefix = output_fname.split('.')[0]
            video_file = visualize_adversary(output_h5, logger.get_snapshot_dir(), output_prefix, **self.video_params)
            save_video_file(output_h5, video_file)

        logger.record_tabular("AverageReturn:", avg_return_adversary)
        save_performance_to_all(all_output_h5, avg_return_adversary, adv_params, len(paths))
        logger.dump_tabular(with_prefix=False)

    def run(self):
        all_output_h5 = init_all_output_file(logger.get_snapshot_dir(), self.adversary_algo,
                                             self.adversary_algo_param_names, self.batch_size)

        if self.policy_adv is not None and self.policy_target is not None:
            assert len(self.norms) == 1 and len(self.games) == 1
            game = self.games[0]
            norm = self.norms[0]
            exp_index_adv = self.policy_adv.split('_')[0]
            algo_name_adv = EXP_IDX_TO_NAME[exp_index_adv].split('_')[0]

            exp_index_target = self.policy_target.split('_')[0]
            algo_name_target = EXP_IDX_TO_NAME[exp_index_target].split('_')[0]

            params_dir_adv = osp.join(self.model_dir, algo_name_adv, exp_index_adv+'-'+game, self.policy_adv)
            params_dir_target = osp.join(self.model_dir, algo_name_target, exp_index_target+'-'+game, self.policy_target)
            params_file_adv = get_latest_param_file(params_dir_adv)
            params_file_target = get_latest_param_file(params_dir_target)

            for fgsm_eps in self.fgsm_eps:
                algo_adv, env_adv = load_model(params_file_adv, self.batch_size)
                algo_target, env_target = load_model(params_file_target, self.batch_size)

                policy_adv = (algo_adv, env_adv, None, self.policy_adv)
                policy_target = (algo_target, env_target, None, self.policy_target)

                self.run_for_adv_target(game, policy_adv, policy_target, norm, fgsm_eps, algo_name_adv, all_output_h5)
            return

        policies = load_models(self.games, self.experiments, self.model_dir, \
                               self.batch_size, self.threshold_perf, self.threshold_n)
        
        
        all_norm_eps = list(itertools.product(self.norms, self.fgsm_eps))
        for norm_eps in all_norm_eps:
            norm = norm_eps[0]
            fgsm_eps = norm_eps[1]
            for game in policies:
                game_policies = []
                for algo_name in policies[game]:
                    game_policies += policies[game][algo_name]
                all_policy_adv_target_idx = list(itertools.product(range(len(game_policies)), \
                                                                   range(len(game_policies))))
                for policy_adv_target_idx in all_policy_adv_target_idx:
                    policy_adv = game_policies[policy_adv_target_idx[0]]
                    policy_target = game_policies[policy_adv_target_idx[1]]

                    # Set up adversary function
                    policy_adv[1].set_adversary_fn(None)
                    if not self.test_transfer and policy_target[3] != policy_adv[3]:
                        continue

                    self.run_for_adv_target(game, policy_adv, policy_target, norm, fgsm_eps, algo_name)

