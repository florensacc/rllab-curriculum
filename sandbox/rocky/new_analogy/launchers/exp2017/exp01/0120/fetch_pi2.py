from gym.spaces import prng

def run_task(*_):
    from gpr import Trajectory
    import gpr.trajectory
    from sandbox.rocky.new_analogy.gpr_ext.pi2_fast import pi2
    from sandbox.rocky.new_analogy import fetch_utils

    gpr.trajectory.optimizers["pi2_fast"] = pi2

    from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics

    horizon = 300

    expr = fetch_utils.gpr_fetch_expr(horizon=horizon, usage="pi2")
    gpr_env = fetch_utils.gpr_fetch_env(horizon=horizon, usage="pi2")

    ffd = FastForwardDynamics(gpr_env)

    from gpr.worldgen.world import set_in_pi2
    set_in_pi2()

    optimizer_params = expr.optimizer_params._replace(
        # optimizer="pi2_fast",
        save_intermediate=True,
        # mpc_horizon=20,
        # mpc_steps=10,
        # skip=1,
        # particles=1000,
        # init_cov=1.,
        # max_kl=100,
        # num_iterations=10,
        extras=dict(ffd=ffd),
    )

    gpr_env.seed(0)
    prng.seed(0)
    trajectory = Trajectory(gpr_env)
    trajectory.optimize(optimizer_params)


run_task()


# run_local(run_task)
