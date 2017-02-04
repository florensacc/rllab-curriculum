from rllab.misc.instrument import run_experiment_lite
from rllab import config

MODE = "local_docker"
# MODE = "ec2"

if MODE in ["local_docker"]:
    n_parallel = 16  # 16#16#2#16#4#1#16
    epoch_length = 100  # 10000#10000#1000#10000#10000
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
    # from sandbox.rocky.chainer.actor_critics.nips_dqn_head_actor_critic import NIPSDQNHeadActorCritic
    # from sandbox.rocky.chainer.actor_critics.custom_atari_actor_critic import CustomAtariActorCritic
    from sandbox.rocky.chainer.algos_ref import ale
    # a3c_ale import A3CALE#a3c import A3C
    import atari_py
    from sandbox.rocky.chainer.algos_ref import rmsprop_async
    from sandbox.rocky.chainer.algos_ref.a3c_ale import A3CFF
    from sandbox.rocky.chainer.algos_ref import async
    from sandbox.rocky.chainer.algos_ref import a3c
    from sandbox.rocky.chainer.algos_ref import dqn_phi
    from sandbox.rocky.chainer.algos_ref.a3c_ale import train_loop
    import chainer
    import time
    import multiprocessing as mp
    import numpy as np

    rom_filename = atari_py.get_game_path("pong")

    n_actions = ale.ALE(rom_filename).number_of_actions

    def model_opt():
        model = A3CFF(n_actions)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return model, opt

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    def run_func(process_idx):
        env = ale.ALE(rom_filename, use_sdl=False)
        # env = ale.ALE(args.rom, use_sdl=args.use_sdl)
        model, opt = model_opt()
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        agent = a3c.A3C(model, opt, t_max=5, gamma=0.99, beta=1e-2,
                        process_idx=process_idx, phi=dqn_phi.dqn_phi)

        from rllab.misc.ext import AttrDict
        train_loop(
            process_idx,
            counter,
            max_score,
            AttrDict(
                steps=8 * 10 ** 7,
                lr=7e-4,
                eval_frequency=10 ** 6,
                eval_n_runs=10,
                rom=rom_filename,
            ),
            agent,
            env,
            start_time
        )

    async.run_async(n_parallel, run_func)

    # import numpy as np
    #
    # # env = AtariEnv(
    # #     game="pong",
    # #     img_width=84,
    # #     img_height=84,
    # #     obs_type="image",
    # #     frame_skip=4,
    # #     max_start_nullops=30,
    # #     avoid_life_lost=True,
    # # )
    #
    #
    # env = AtariEnv(
    #     rom_filename=rom_filename,  # game="pong",
    #     # rom_filename=os.path.join(rom_dir,game+".bin"),
    #     # plot=plot,
    #     # record_ram=record_ram,
    # )
    # # env = AtariEnv(
    # #     game="pong",
    # #     img_width=84,
    # #     img_height=84,
    # #     obs_type="image",
    # #     frame_skip=4,
    # #     max_start_nullops=30,
    # #     avoid_life_lost=True,
    # # )
    #
    # agent = A3CAgent(
    #     n_actions=env.number_of_actions,
    #     beta=0.01,  # entropy_bonus,
    #     sync_t_gap_limit=sync_t_gap_limit,
    #
    # )
    #
    # policy = NIPSDQNHeadActorCritic(
    #     env_spec=env.spec,
    # )
    # # policy = CustomAtariActorCritic(
    # #     env_spec=env.spec,
    # # )
    #
    # algo = A3C(
    #     env=env,
    #     policy=policy,
    #     n_envs_per_worker=1,
    #     update_frequency=5,
    #     learning_rate=7e-4,
    #     max_grad_norm=40.0,
    #     n_parallel=n_parallel,
    #     max_path_length=np.inf,
    #     discount=0.99,
    #     gae_lambda=1.,
    #     epoch_length=epoch_length,
    #     optimizer_type="rmsprop_async",
    #     n_epochs=2000,
    #     scale_reward=1.,
    #     normalize_adv=False,
    #     normalize_vf=False,
    #     normalize_momentum=0,
    #     entropy_bonus_coeff=0.01,
    #     share_optimizer=False,
    # )
    #
    # algo.train()


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
