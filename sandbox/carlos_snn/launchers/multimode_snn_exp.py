


from sandbox.carlos_snn.envs.multiMod2D_env import MultiModEnv
from sandbox.carlos_snn.envs.bimod_env_tunable import BimodEnv

from sandbox.rocky.snn.baselines.linear_feature_snn_baseline import LinearFeatureSNNBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from sandbox.rocky.snn.policies.stochastic_gaussian_mlp_policy import StochasticGaussianMLPPolicy
from sandbox.rocky.snn.algos.snn_algos import TRPO_snn
from sandbox.rocky.snn.hallucinators.prior_hallucinator import PriorHallucinator
from rllab.misc.instrument import stub, run_experiment_lite

import datetime
import dateutil.tz
now = datetime.datetime.now(dateutil.tz.tzlocal())

stub(globals())

for eps in [0.01]:
    for disp in [0.5]:
        # env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=True)
        # env = MultiModEnv(mu=[1,0], sigma=0.01, n=2, rand_init=False)

        # baseline1 = [LinearFeatureSNNBaseline(env_spec=env.spec), 'SNNbaseline']
        # baseline2 = [LinearFeatureBaseline(env_spec=env.spec), 'baseline']

        env = BimodEnv(eps=eps, disp=disp, mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)
        baseline = [LinearFeatureBaseline(env_spec=env.spec), 'baseline']

        for base in [baseline]:
            for latent_dim in [0, 1, 2]:
                for n_samples in [5]:
                    # for hid_latent in [1, 2, 5]:
                        if latent_dim == 0:
                            latents_def = None
                        else:
                            latents_def = [('independent_gaussian', latent_dim)]
                        policy = StochasticGaussianMLPPolicy(
                            env_spec=env.spec,
                            input_latent_vars=latents_def,
                            hidden_latent_vars=[
                                # [('independent_gaussian', hid_latent)],
                                # [('independent_gaussian', hid_latent)],
                                [],
                                [],
                            ],
                            hidden_sizes=(8, 8)
                        )

                        baseline = base[0]

                        algo = TRPO_snn(
                            env=env,
                            policy=policy,
                            baseline=baseline,
                            # self_normalize=False,
                            hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                            # hallucinator=PosteriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                            batch_size=500,
                            whole_paths=True,
                            max_path_length=100,
                            n_itr=100,
                            discount=0.99,
                            step_size=0.01,
                        )

                        for s in [4,5,155,21,56,13]:
                            now = datetime.datetime.now(dateutil.tz.tzlocal())
                            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

                            run_experiment_lite(
                                stub_method_call=algo.train(),
                                n_parallel=1,
                                snapshot_mode="last",
                                seed=s,
                                exp_prefix='SNNtrpo-local-opt',
                                exp_name='SNNtrpo-local-opt_eps{}_disp{}_latGau{}_hallu{}_s{:04d}_{}'.format(
                                    ''.join(str(eps).split('.')), ''.join(str(disp).split('.')),
                                    latent_dim, n_samples, s, timestamp),
                            )
