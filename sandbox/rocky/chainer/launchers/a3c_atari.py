import pickle

from rllab.misc.instrument import run_experiment_lite
from rllab import config

MODE = "local_docker"
# MODE = "ec2"

if MODE in ["local_docker"]:
    n_parallel = 16  # 16#16#16#16#16#2#16#4#1#16
    epoch_length = 10000  # 10000#10000#1000#10000#10000
else:
    n_parallel = 16
    epoch_length = 50000

if n_parallel == 1:
    env = dict()
else:
    env = dict(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )


def run_task(*_):
    from sandbox.rocky.chainer.actor_critics.nips_dqn_head_actor_critic import NIPSDQNHeadActorCritic
    from sandbox.rocky.chainer.actor_critics.custom_atari_actor_critic import CustomAtariActorCritic
    from sandbox.rocky.chainer.algos.a3c import A3C
    from sandbox.rocky.chainer.envs.atari_env import AtariEnv
    import numpy as np

    env = AtariEnv(
        game="pong",
        img_width=84,
        img_height=84,
        obs_type="image",
        frame_skip=4,
        max_start_nullops=30,
        avoid_life_lost=True,
    )

    policy = NIPSDQNHeadActorCritic(
        env_spec=env.spec,
    )

    from sandbox.rocky.chainer.optimizers.rmsprop_async import RMSpropAsync
    optimizer = RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
    import chainer

    optimizer.setup(policy)
    optimizer.add_hook(chainer.optimizer.GradientClipping(40.))

    from sandbox.rocky.chainer.algos_ref import async
    shared_params = async.share_params_as_shared_arrays(policy)
    shared_states = async.share_states_as_shared_arrays(optimizer)

    algo = A3C(
        env=env,
        policy=policy,
        shared_params=shared_params,
        shared_states=shared_states,
        update_frequency=5,
        optimizer=optimizer,
        n_parallel=n_parallel,
        max_path_length=np.inf,
        discount=0.99,
        gae_lambda=1.,
        epoch_length=epoch_length,
        n_epochs=2000,
        scale_reward=1.,
        normalize_adv=False,
        normalize_vf=False,
        normalize_momentum=0,
        entropy_bonus_coeff=0.01,
    )

    algo.train()


config.AWS_INSTANCE_TYPE = "c4.8xlarge"
config.AWS_SPOT = True
config.AWS_SPOT_PRICE = '1.0'
config.AWS_REGION_NAME = 'us-west-1'
config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

run_experiment_lite(
    run_task,
    mode=MODE,
    env=env,
    exp_prefix="a3c-atari",
    use_cloudpickle=True,
    docker_image="dementrock/rllab3-shared-gpu-cuda80:20161120",
    terminate_machine=False,
)
