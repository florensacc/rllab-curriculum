from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.neural_learner.algos.saac import ImageJointActorCritic
from sandbox.rocky.neural_learner.policies.greedy_softmax_policy import GreedySoftmaxPolicy

"""
Benchmark on Atari, with image obs
"""

USE_GPU = False  # True
USE_CIRRASCALE = False  # True
# MODE = "local"

MODE = "ec2"
# MODE = launch_cirrascale#()
OPT_BATCH_SIZE = 256
OPT_N_STEPS = 32  # 128


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]  # , 41, 51]

    @variant
    def batch_size(self):
        if MODE == "local":
            return [10000]
        return [50000]

    @variant
    def docker_image(self):
        return [
            "dementrock/rllab3-shared",
        ]

    @variant
    def nonlinearity(self):
        return ["relu"]

    @variant
    def game(self):
        return ["breakout", "beam_rider", "pong", "qbert"]#"breakout", "qbert"]

    @variant
    def learning_rate(self):
        return [7e-4]#, 1e-3]  # 1e-2, 5e-3, 1e-3, 7e-4]

    @variant
    def algo(self):
        return ["saac"]

    @variant
    def t_max(self):
        return [5, 10, 20]

    @variant
    def network(self):
        return ["ff"]  # , "rnn"]#, "ff"]

    @variant
    def hidden_sizes(self):
        return [(32, 32)]  # , (32, 32), (64, 64), (128, 128), (256, 256)]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:

    def run_task(v):
        from sandbox.rocky.neural_learner.envs.parallel_atari_env import AtariEnv
        from sandbox.rocky.neural_learner.algos.pposgd_clip_ratio import PPOSGD
        from sandbox.rocky.neural_learner.algos.saac import SAAC
        from sandbox.rocky.neural_learner.algos.saac import JointActorCritic
        from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
        from sandbox.rocky.neural_learner.policies.categorical_rnn_policy import CategoricalRNNPolicy
        from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        from sandbox.rocky.neural_learner.optimizers.sgd_optimizer import SGDOptimizer
        from sandbox.rocky.tf.algos.trpo import TRPO
        from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
        from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, \
            FiniteDifferenceHvp

        import tensorflow as tf

        from sandbox.rocky.tf.policies.rnn_utils import NetworkType

        env = TfEnv(
            AtariEnv(game=v["game"], terminate_on_life_lost=True, reset_on_life_lost=False, obs_type="image"))
        eval_env = TfEnv(
            AtariEnv(game=v["game"], terminate_on_life_lost=False, reset_on_life_lost=False, obs_type="image"))

        if v["algo"] == "saac":
            actor_critic = ImageJointActorCritic(env_spec=env.spec)
            policy = actor_critic.policy
            vf = actor_critic.vf
            algo = SAAC(
                env=env,
                policy=policy,
                vf=vf,
                n_envs=32,
                t_max=v["t_max"],
                epoch_length=50000,
                n_epochs=1000,
                clip_rewards=(-1, 1),
                learning_rate=v["learning_rate"],
                post_evals=[
                    dict(
                        env=eval_env,
                        policy=policy,
                        batch_size=5,
                        max_path_length=4500,
                        n_envs=5,
                        label="Eval",
                    ),
                    dict(
                        env=eval_env,
                        policy=GreedySoftmaxPolicy(policy),
                        batch_size=5,
                        max_path_length=4500,
                        n_envs=5,
                        label="EvalGreedy",
                    ),
                ]
            )
        else:
            baseline = LinearFeatureBaseline(
                env_spec=env.spec
            )
            if v["network"] == "rnn":
                policy = CategoricalRNNPolicy(
                    env_spec=env.spec,
                    hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
                    weight_normalization=True,
                    network_type=NetworkType.GRU,
                    state_include_action=False,
                    name="policy"
                )
            elif v["network"] == "ff":
                policy = CategoricalMLPPolicy(
                    env_spec=env.spec,
                    hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
                    hidden_sizes=(32, 32),
                    name="policy"
                )
                pass
            else:
                raise NotImplementedError

            if v["algo"] == "pposgd":

                if v["network"] == "rnn":
                    optimizer = TBPTTOptimizer(
                        batch_size=OPT_BATCH_SIZE,
                        n_steps=OPT_N_STEPS,
                        n_epochs=v["min_epochs"],
                    )
                elif v["network"] == "ff":
                    optimizer = SGDOptimizer(
                        batch_size=OPT_BATCH_SIZE,
                        n_epochs=v["min_epochs"],
                    )
                else:
                    raise NotImplementedError

                algo = PPOSGD(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=v["batch_size"],
                    max_path_length=4500,
                    # n_vectorized_envs=0,
                    n_itr=1000,
                    clip_lr=v["clip_lr"],
                    log_loss_kl_before=False,
                    log_loss_kl_after=False,
                    use_kl_penalty=v["use_kl_penalty"],
                    min_n_epochs=v["min_epochs"],
                    optimizer=optimizer,
                    use_line_search=True,
                    entropy_bonus_coeff=v["entropy_bonus_coeff"],
                    backtrack_ratio=0.8,
                )
            elif v["algo"] == "trpo":
                if v["network"] == "rnn":
                    optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
                elif v["network"] == "ff":
                    optimizer = ConjugateGradientOptimizer()
                else:
                    raise NotImplementedError
                algo = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=v["batch_size"],
                    max_path_length=4500,
                    n_itr=1000,
                    optimizer=optimizer,
                )
            else:
                raise NotImplementedError
        algo.train()


    config.DOCKER_IMAGE = vv["docker_image"]
    config.KUBE_DEFAULT_NODE_SELECTOR = {
        "aws/type": "c4.8xlarge",
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 36 * 0.75,
            "memory": "50Gi",
        },
    }
    config.AWS_INSTANCE_TYPE = "m4.2xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = 'us-west-1'
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="1")
    else:
        env = dict()

    # run_task(vv)

    run_experiment_lite(
        run_task,
        exp_prefix="saac-test-5-3",
        mode=MODE,
        n_parallel=0,
        seed=vv["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        variant=vv,
        snapshot_mode="last",
        env=env,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        # python_command="kernprof -l",
    )
    # sys.exit()
