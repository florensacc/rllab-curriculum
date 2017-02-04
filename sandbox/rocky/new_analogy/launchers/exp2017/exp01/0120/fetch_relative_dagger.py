from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    import tensorflow as tf
    import numpy as np

    horizon = 300

    gpr_env = fetch_utils.gpr_fetch_env(horizon=horizon)
    env = fetch_utils.fetch_env(seed=vv["env_seed"], horizon=horizon)
    demo_policy = fetch_utils.fetch_prescribed_policy(gpr_env=gpr_env)

    policy = fetch_utils.FetchWrapperPolicy(
        env_spec=env.spec,
        wrapped_policy=GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(256, 256),
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )
    )

    path = fetch_utils.demo_path(seed=vv["env_seed"], env=env, policy=demo_policy)

    batch_size = 512
    n_updates_per_epoch = 1000

    trainer = fetch_utils.bc_trainer(env=env, policy=policy)
    trainer.add_paths([path])

    eval_policy = fetch_utils.DeterministicPolicy(env_spec=env.spec, wrapped_policy=policy)

    for _ in trainer.train_loop(batch_size=batch_size, n_updates_per_epoch=n_updates_per_epoch):
        logger.log("Sampling new trajectory")
        path = rollout(env, eval_policy, max_path_length=horizon)
        logger.record_tabular('FinalReward', path["rewards"][-1])
        logger.record_tabular('TotalReward', np.sum(path["rewards"]))
        logger.record_tabular('IsSuccess', int(env.wrapped_env._is_success(path)))
        # compute optimal actions
        logger.log("Annotating new trajectory")
        obs = env.observation_space.unflatten_n(path["observations"])
        actions = []
        for x, ob in zip(path["env_infos"]["x"], obs):
            gpr_env.reset_to(x)
            demo_policy.env = gpr_env
            actions.append(demo_policy.get_action(ob)[0])
        path["actions"] = np.asarray(actions)
        trainer.add_paths([path])
        logger.dump_tabular()


for env_seed in range(10):
    run_cirrascale(
    # run_local_docker(
        run_task,
        exp_name="fetch-relative-dagger-1",
        variant=dict(env_seed=env_seed),
        seed=env_seed,
    )
    # break
