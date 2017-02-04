from gpr.runner import TrajectoryRunner
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_local


def run_task(*_):
    from gpr import Trajectory
    import gpr.trajectory
    from sandbox.rocky.new_analogy.gpr_ext.pi2_fast import pi2
    from sandbox.rocky.new_analogy import fetch_utils
    import numpy as np

    env = fetch_utils.fetch_env(seed=0)
    expr = fetch_utils.gpr_fetch_expr()

    path = fetch_utils.demo_traj(seed=0, env=env)

    # while True:
    #     for xid in range(0, len(path["observations"]), 50):
    # while True:
    #     xid = 50
    #     x = path["env_infos"]["x"][xid]
    #     env.wrapped_env.gpr_env.reset_to(x)
    #     env.render()
    #     import time
    #     time.sleep(0.1)
    # worker_id = vv["worker_id"]

    # from gpr.envs.stack import Experiment
    # expr = Experiment(nboxes=2, horizon=1000)
    # env = expr.make(task_id=task_id)
    # import gpr.trajectory

    # import ipdb;
    # ipdb.set_trace()
    gpr.trajectory.optimizers["pi2_fast"] = pi2

    from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics

    # target = path["env_infos"]["site_xpos"][50]
    target = [None]

    lookahead = 10

    gpr_env = fetch_utils.gpr_fetch_env(mocap=True, horizon=lookahead)


    def custom_reward(xprev, xnext, sense, mjparallel, t, u):
        diff = xnext - target
        l2 = np.sum(np.square(diff), axis=-1)
        l1 = np.sum(np.abs(diff), axis=-1)
        lhalf = np.sum(np.sqrt(np.abs(diff)), axis=-1)
        reward = -l2 - l1 - lhalf
        ret = reward
        return ret

    ffd = FastForwardDynamics(gpr_env, custom_reward=custom_reward, custom_specs={"use_site_xpos"})

    optimizer_params = expr.optimizer_params._replace(
        optimizer="pi2_fast",
        save_intermediate=False,
        mpc_horizon=lookahead,
        mpc_steps=lookahead,
        skip=1,
        particles=1000,
        init_cov=3.,
        max_kl=100,
        num_iterations=10,
        extras=dict(ffd=ffd),
    )

    gpr_env.reset()

    x = path["env_infos"]["x"][0]

    for t in range(1000):
        target[0] = path["env_infos"]["x"][t + lookahead]

        import gym.spaces.prng

        gpr_env.seed(0)
        gym.spaces.prng.seed(0)
        trajectory = Trajectory(gpr_env)
        trajectory.xinit = x  # path["env_infos"]["x"][t]
        trajectory.optimize(optimizer_params)

        gpr_env.x = x
        for u in trajectory.solution['u']:
            gpr_env.step(u)
        x = gpr_env.x
        # TrajectoryRunner([trajectory]).run()
        # while True:
        gpr_env.render()
            # import time;
            # time.sleep(0.001)
        import ipdb; ipdb.set_trace()


run_task()


# run_local(run_task)
