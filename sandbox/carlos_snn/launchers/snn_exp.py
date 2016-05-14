from __future__ import print_function
from __future__ import absolute_import

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.snn.baselines.linear_feature_snn_baseline import LinearFeatureSNNBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
# from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.gym_env import GymEnv

from rllab.envs.normalized_env import normalize

from sandbox.carlos_snn.policies.stochastic_gaussian_mlp_policy import StochasticGaussianMLPPolicy

from sandbox.rocky.snn.algos.snn_algos import TRPO_snn
from sandbox.rocky.snn.hallucinators.prior_hallucinator import PriorHallucinator
from sandbox.rocky.snn.hallucinators.posterior_hallucinator import PosteriorHallucinator
from rllab.misc.instrument import stub, run_experiment_lite

import datetime
import dateutil.tz
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

stub(globals())


env_name='Hopper-v1'
env = normalize(GymEnv(env_name))

# baseline1 = [LinearFeatureSNNBaseline(env_spec=env.spec),'SNNbaseline']
baseline2 = [LinearFeatureBaseline(env_spec=env.spec),'baseline']

for base in [baseline2]:
    for latent_dim in [0]:#[0,1,2,5,11]:
        for n_samples in [0,1,5,10]:
            # for hid_latent in [1, 2, 5]:
                if latent_dim == 0:
                    latents_def = None
                else:
                    latents_def = [('independent_bernoulli', latent_dim)]
                policy = StochasticGaussianMLPPolicy(
                    env_spec=env.spec,
                    input_latent_vars=latents_def,
                    hidden_latent_vars=[
                        # [('independent_gaussian', hid_latent)],
                        # [('independent_gaussian', hid_latent)],
                        [],
                        [],
                    ],
                    hidden_sizes=(32, 32)
                )

                baseline = base[0]

                algo = TRPO_snn(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    self_normalize=True,
                    # hallucinator = None,
                    hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                    # hallucinator=PosteriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                    batch_size=2000,
                    whole_paths=True,
                    max_path_length=100,
                    n_itr=3,
                    discount=0.99,
                    step_size=0.01,
                )

                for s in [4]:
                    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
                    run_experiment_lite(
                        stub_method_call=algo.train(),
                        n_parallel=1,
                        snapshot_mode="last",
                        seed=s,
                        exp_prefix=env_name + '_test',
                        exp_name='{}_{}_{}Blatent_{}halluPrior_{:04d}'.format(
                            env_name,base[1], latent_dim, n_samples, s),
                    )
                    # sys.exit(0)
