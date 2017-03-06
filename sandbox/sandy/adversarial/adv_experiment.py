"""AdvExperiment class runs rollouts of a policy with adversarial perturbations
"""

import itertools, os.path as osp
import numpy as np

from rllab.misc import logger

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation
from sandbox.sandy.adversarial.fgsm_sleeper_dualdescent import fgsm_sleeper_perturbation
from sandbox.sandy.adversarial.io_util import init_output_file, \
        save_performance, save_video_file, init_all_output_file, save_performance_to_all
from sandbox.sandy.adversarial.vis import visualize_adversary
from sandbox.sandy.misc.util import get_time_stamp, to_iterable
from sandbox.sandy.shared.model import TrainedModel, load_models, model_dir_to_exp_name
from sandbox.sandy.shared.model_rollout import get_average_return
from sandbox.sandy.shared.experiment import get_experiments

DEFAULT_OBS_MIN = 0
DEFAULT_OBS_MAX = 1
DEFAULT_THRESHOLD_PERF = 0.8
DEFAULT_THRESHOLD_N = 3
DEFAULT_N = 10

class AdvExperiment(object):
    def __init__(self, games, norms, fgsm_eps, exp_names, base_dir, \
                 save_rollouts, test_transfer, adversary_algo, \
                 adversary_algo_param_names, seed, **kwargs):
        self.games = to_iterable(games)
        self.norms = to_iterable(norms)
        self.fgsm_eps = to_iterable(fgsm_eps)
        self.exp_names = to_iterable(exp_names)
        self.adv_model_dir = kwargs.get("adv_model_dir", None)
        self.target_model_dir = kwargs.get("target_model_dir", None)

        if self.games is None or self.exp_names is None:
            assert self.adv_model_dir is not None and self.target_model_dir is not None
        self.base_dir = base_dir
        self.save_rollouts = save_rollouts
        self.test_transfer = test_transfer  # If False, only runs exp_names where adversarial policy == target policy
        self.adversary_algo = adversary_algo
        self.adversary_algo_param_names = adversary_algo_param_names
        self.seed = seed
        self.obs_min = kwargs.get('obs_min', DEFAULT_OBS_MIN)
        self.obs_max = kwargs.get('obs_max', DEFAULT_OBS_MAX)
        self.threshold_perf = kwargs.get('threshold_perf', DEFAULT_THRESHOLD_PERF)
        self.threshold_n = kwargs.get('threshold_n', DEFAULT_THRESHOLD_N)
        self.N = kwargs.get('N', DEFAULT_N)
        self.video_params = kwargs.get("video_params", {})

    def log_run_info(self, policy_adv, policy_target, variant):
        logger.record_tabular("Game", policy_adv.exp.game)
        logger.record_tabular("AdversaryPolicy", policy_adv.model_name)
        logger.record_tabular("TargetPolicy", policy_target.model_name)
        for k in variant:
            if k == "norm":
                key = "Norm"
            elif k == "fgsm_eps":
                key = "FGSMEps"
            else:
                key = k
            logger.record_tabular(key, variant[k])

    def get_output_h5(self, policy_adv, policy_target, variant, algo_params):
        output_fname = "{algo_name}_{norm}_{eps}_{policy_adv}_{policy_target}.h5".format(
            algo_name=policy_adv.exp.algo_name,
            norm=variant['norm'],
            eps=str(variant['fgsm_eps']).replace('.', '-'),
            policy_adv=policy_adv.model_name,
            policy_target=policy_target.model_name
        )
        output_h5 = init_output_file(logger.get_snapshot_dir(), \
                                     None, 'fgsm', algo_params, \
                                     fname=output_fname, \
                                     algo_name=policy_adv.exp.algo_name)
        return output_h5

    def run_for_adv_target(self, policy_adv, policy_target, variant, all_output_h5):
        variant = self.variant_to_dict(variant)
        self.log_run_info(policy_adv, policy_target, variant)

        output_h5 = None
        if self.save_rollouts:
            algo_params = {'eps': variant["fgsm_eps"], 'norm': variant["norm"]}
            output_h5 = self.get_output_h5(policy_adv, policy_target, variant, algo_params)

        if self.adversary_algo == 'fgsm':
            adv_params = dict(
                    fgsm_eps=variant["fgsm_eps"],
                    norm=variant["norm"],
                    obs_min=self.obs_min,
                    obs_max=self.obs_max,
                    output_h5=output_h5,
                    policy_adv=policy_adv.model_name,
                    policy_rollout=policy_target.model_name
            )
<<<<<<< HEAD
            adversary_fn = lambda x: fgsm_perturbation(x, policy_adv.algo, **adv_params)
=======
            adversary_fn = lambda x, y: fgsm_perturbation(x, y, policy_adv.algo, **adv_params)
>>>>>>> upstream/master
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

        logger.record_tabular("AverageReturn", avg_return_adversary)
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
                self.run_for_adv_target(policy_adv, policy_target, (norm, fgsm_eps), \
                                        all_output_h5)
            return

        print("LOADING MODELS:", self.norms, self.fgsm_eps)
        adv_policies = load_models(self.games, self.exp_names, self.base_dir, \
                                   self.threshold_perf, self.threshold_n)
        target_policies = load_models(self.games, self.exp_names, self.base_dir, \
                                      self.threshold_perf, self.threshold_n)
        for variant in self.variants:
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
                                                variant, all_output_h5)
    @property
    def variants(self):
        return list(itertools.product(self.norms, self.fgsm_eps))

    def variant_to_dict(self, variant):
        return dict(norm=variant[0], fgsm_eps=variant[1])

class AdvSleeperExperiment(AdvExperiment):
    def __init__(self, games, norms, fgsm_eps, exp_names, base_dir, \
                 save_rollouts, test_transfer, adversary_algo, \
                 adversary_algo_param_names, seed, **kwargs):
        AdvExperiment.__init__(self, games, norms, fgsm_eps, exp_names, base_dir, \
                 save_rollouts, test_transfer, adversary_algo, \
                 adversary_algo_param_names, seed, **kwargs)
        self.num_ts = kwargs.get("num_ts", 100)
        # across_rollout - if True, sleeper adversary introduces perturbations
        #     across the whole rollout, rather than specifically (and separately)
        #     at each of the pre-computed timesteps. When this is True, num_ts
        #     is ignored
        self.across_rollout = kwargs.get("across_rollout", False)
        self.k_init_lambda = to_iterable(kwargs.get("k_init_lambda", [(1,None)]))
        self.dual_descent_stepsizes = kwargs.get("dual_descent_stepsizes", None)
        if self.dual_descent_stepsizes is not None:
            self.dual_descent_stepsizes = to_iterable(self.dual_descent_stepsizes)
        self.deterministic = True
        self.save_rollouts = True  # Overrides input parameter

        print("Variants:", self.variants)

    @property
    def variants(self):
        return list(itertools.product(self.norms, self.fgsm_eps, self.k_init_lambda))

    def variant_to_dict(self, variant):
        init_lambda = variant[2][1]
        if init_lambda is not None:
            init_lambda = np.array(init_lambda)
        return dict(norm=variant[0], fgsm_eps=variant[1], k=variant[2][0], \
                    init_lambda=init_lambda)

    def run_for_adv_target(self, policy_adv, policy_target, variant, all_output_h5, \
                           set_frame_dropout_zero = True):
        # k - number of time steps of delay for adversarial perturbation; i.e.,
        #     perturbation happens at time t, target policy acts normally for times
        #     t through t+k-1, does different action at time t+k
        # ts - time steps for adversary to make perturbation at; can either be
        #     integers (corresponding to actual time steps), or floats between 0 and
        #     1 (fraction of the total trajectory length). If None, num_ts timesteps
        #     are taken at even intervals along the rollout
        # deterministic - if True, target policy always picks argmax action to
        #     execute> If False, policy sampled from output distribution over actions.

        if set_frame_dropout_zero:
            policy_adv.env.frame_dropout = 0
            policy_target.env.frame_dropout = 0

        variant = self.variant_to_dict(variant)
        self.log_run_info(policy_adv, policy_target, variant)



        # Do complete rollout to figure out its length
        policy_target.env.set_adversary_fn(None)
        policy_target.algo.cur_agent.model.lstm.reset_state()
        avg_return_noadv, paths, timesteps = \
                get_average_return(policy_target.algo, self.seed, N=1, \
                                   return_timesteps=True, \
                                   deterministic=self.deterministic, \
                                   check_equiv=True)
        logger.record_tabular("AvgReturnNoAdv", avg_return_noadv)
        logger.record_tabular("TimestepsNoAdv", timesteps)

        # Precompute time step t's to try sleeper perturbations at
        if not self.across_rollout:
            ts = timesteps * np.linspace(0,1,self.num_ts,endpoint=False)
            ts = np.round(ts).astype(int)
            # Get rid of duplicates
            ts = sorted(list(set(ts)))
        else:
            ts = [-1]

        output_h5 = None
        if self.save_rollouts:
            algo_params = {'eps': variant["fgsm_eps"], 'norm': variant["norm"], \
                           'k': variant["k"]}
            if not self.across_rollout:
                algo_params['ts'] = ts
            output_h5 = self.get_output_h5(policy_adv, policy_target, variant, algo_params)

        # If self.across_rollouts is True:
        # For each t, run policy rollout (without any adversarial perturbations)
        # until that time step, then perturb at time t and branch into 
        # two rollouts (for current and next k timesteps, i.e., time steps t through
        # t+k) after perturbing at time t vs. rollout without perturbing
        # at time t; save the two sequences of output action distributions
        # (to output_h5) to see if indeed only the last one (at time step t+k) is changed
        # 
        # If self.across_rollouts is False:
        # Starting at time t = 0, try to find a valid adversarial sleeper
        # perturbation at that time step (i.e., changes the agent's actions
        # at time step t+k but not at time steps t through t+k-1). If no valid
        # perturbation is found, don't perturb and move on to the next time step.
        # Also don't perturb if a sleeper perturbation was already introduced
        # within the last k timesteps.
        assert type(policy_adv.algo).__name__ == "A3CALE"
        policy_adv.algo.cur_agent.model.skip_unchain = True
        logger.record_tabular("ts", ts)

        for t in ts:
            policy_target.algo.cur_agent.model.lstm.reset_state()
            policy_adv.algo.cur_agent.model.lstm.reset_state()

            if self.adversary_algo == 'fgsm':
                adv_params = dict(
                        k=variant["k"],
                        fgsm_eps=variant["fgsm_eps"],
                        norm=variant["norm"],
                        obs_min=self.obs_min,
                        obs_max=self.obs_max,
                        output_h5=output_h5,
                        policy_adv=policy_adv.model_name,
                        policy_rollout=policy_target.model_name,
                        dual_descent_stepsizes=self.dual_descent_stepsizes,
                        init_lambda = variant["init_lambda"],
                        across_rollout=self.across_rollout
                )
                if not self.across_rollout:
                    adv_params['t'] = t
                adversary_fn = lambda x,y: fgsm_sleeper_perturbation(x, y, policy_adv.algo, **adv_params)
            else:
                raise NotImplementedError
            policy_target.env.set_adversary_fn(adversary_fn)

            avg_return_adv, paths, timesteps = \
                    get_average_return(policy_target.algo, self.seed, N=self.N, \
                                       return_timesteps=True, \
                                       deterministic=self.deterministic, \
                                       check_equiv=True)
            logger.record_tabular('Timesteps', timesteps)
            logger.record_tabular("AverageReturn:", avg_return_adv)
            save_performance_to_all(all_output_h5, avg_return_adv, adv_params, len(paths), timesteps=timesteps)
            logger.dump_tabular(with_prefix=False)
