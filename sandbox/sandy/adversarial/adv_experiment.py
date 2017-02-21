"""AdvExperiment class runs rollouts of a policy with adversarial perturbations
"""

import itertools, os.path as osp

from rllab.misc import logger

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation
from sandbox.sandy.adversarial.io_util import init_output_file, \
        save_performance, save_video_file, init_all_output_file, save_performance_to_all
from sandbox.sandy.adversarial.vis import visualize_adversary
from sandbox.sandy.misc.util import get_time_stamp, to_iterable
from sandbox.sandy.shared.model import TrainedModel, load_models, model_dir_to_exp_name
from sandbox.sandy.shared.model_rollout import get_average_return
from sandbox.sandy.shared.experiment import get_experiments

class AdvExperiment(object):
    def __init__(self, games, norms, fgsm_eps, exp_names, base_dir, \
                 save_rollouts, test_transfer, adversary_algo, \
                 adversary_algo_param_names, seed, obs_min=0, obs_max=1, \
                 threshold_perf=0.8, threshold_n=3, N=10,
                 video_params={}, adv_model_dir=None, target_model_dir=None):
        self.games = to_iterable(games)
        self.norms = to_iterable(norms)
        self.fgsm_eps = to_iterable(fgsm_eps)
        self.exp_names = to_iterable(exp_names)
        self.adv_model_dir = adv_model_dir
        self.target_model_dir = target_model_dir

        if self.games is None or self.exp_names is None:
            assert self.adv_model_dir is not None and self.target_model_dir is not None
        self.base_dir = base_dir
        self.save_rollouts = save_rollouts
        self.test_transfer = test_transfer  # If False, only runs exp_names where adversarial policy == target policy
        self.adversary_algo = adversary_algo
        self.adversary_algo_param_names = adversary_algo_param_names
        self.seed = seed
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.threshold_perf = threshold_perf
        self.threshold_n = threshold_n
        self.N = N
        self.video_params=video_params

    def run_for_adv_target(self, policy_adv, policy_target, norm, fgsm_eps, all_output_h5):
        logger.record_tabular("Game", policy_adv.exp.game)
        logger.record_tabular("AdversaryPolicy", policy_adv.model_name)
        logger.record_tabular("TargetPolicy", policy_target.model_name)
        logger.record_tabular("Norm", norm)
        logger.record_tabular("FGSMEps", fgsm_eps)
        output_h5 = None
        if self.save_rollouts:
            output_fname = "{algo_name}_{norm}_{eps}_{policy_adv}_{policy_target}.h5".format(
                algo_name=policy_adv.exp.algo_name,
                norm=norm,
                eps=str(fgsm_eps).replace('.', '-'),
                policy_adv=policy_adv.model_name,
                policy_target=policy_target.model_name
            )
            output_h5 = init_output_file(logger.get_snapshot_dir(), \
                                         None, 'fgsm', \
                                         {'eps': fgsm_eps, 'norm': norm}, \
                                         fname=output_fname, \
                                         algo_name=policy_adv.exp.algo_name)

        if self.adversary_algo == 'fgsm':
            adv_params = dict(
                    fgsm_eps=fgsm_eps,
                    norm=norm,
                    obs_min=self.obs_min,
                    obs_max=self.obs_max,
                    output_h5=output_h5,
                    policy_adv=policy_adv.model_name,
                    policy_rollout=policy_target.model_name
            )
            adversary_fn = lambda x, y: fgsm_perturbation(x, y, policy_adv.algo, **adv_params)
        else:
            raise NotImplementedError

        # Run policy rollouts with FGSM adversary for N trials, get average return
        policy_target.env.set_adversary_fn(adversary_fn)
        avg_return_adversary, paths, timesteps = \
                get_average_return(policy_target.algo, self.seed, \
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
                                             self.adversary_algo_param_names)

        if self.adv_model_dir is not None and self.target_model_dir is not None:
            assert len(self.norms) == 1 and len(self.games) == 1
            norm = self.norms[0]
            adv_exp_name = model_dir_to_exp_name(self.adv_model_dir)
            target_exp_name = model_dir_to_exp_name(self.target_model_dir)
            adv_experiment = get_experiments(self.games, [adv_exp_name], self.base_dir)[0]
            target_experiment = get_experiments(self.games, [target_exp_name], self.base_dir)[0]

            policy_adv = TrainedModel(self.adv_model_dir, adv_experiment)
            policy_target = TrainedModel(self.target_model_dir, target_experiment)

            for fgsm_eps in self.fgsm_eps:
                self.run_for_adv_target(policy_adv, policy_target, norm, \
                                        fgsm_eps, all_output_h5)
            return

        print("LOADING MODELS:", self.norms, self.fgsm_eps)
        adv_policies = load_models(self.games, self.exp_names, self.base_dir, \
                                   self.threshold_perf, self.threshold_n)
        target_policies = load_models(self.games, self.exp_names, self.base_dir, \
                                      self.threshold_perf, self.threshold_n)
        all_norm_eps = list(itertools.product(self.norms, self.fgsm_eps))
        for norm_eps in all_norm_eps:
            norm, fgsm_eps = norm_eps
            for game in self.games:
                adv_game_policies = list(itertools.chain.from_iterable(adv_policies[game].values()))
                target_game_policies = list(itertools.chain.from_iterable(target_policies[game].values()))
                print("\t # policies:", len(adv_game_policies), len(target_game_policies))
                # Test out all adversary-target pairs in game_policies
                # (or only white-box ones, if test_transfer = False)
                for policy_adv in adv_game_policies:
                    for policy_target in target_game_policies:
                        # Make sure adversary policy's adversary_fn is set to None
                        policy_adv.env.set_adversary_fn(None)
                        if not self.test_transfer and \
                           policy_target.model_name != policy_adv.model_name:
                            continue
                        self.run_for_adv_target(policy_adv, policy_target, \
                                                norm, fgsm_eps, all_output_h5)

