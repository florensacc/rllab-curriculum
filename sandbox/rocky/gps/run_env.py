


from sandbox.rocky.gps.envs.symbolic_swimmer_env import SymbolicSwimmerEnv
from sandbox.rocky.gps.envs.symbolic_double_pendulum_env import SymbolicDoublePendulumEnv
from sandbox.rocky.gps.envs.symbolic_cartpole_env import SymbolicCartpoleEnv
from sandbox.rocky.gps.envs.symbolic_cartpole_swingup_env import SymbolicCartpoleSwingupEnv
from sandbox.rocky.gps.envs.symbolic_env import SymbolicNormalize
from sandbox.rocky.gps.algos.wrapped_ilqr import WrappedILQR
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

envs = list(map(SymbolicNormalize, [
    SymbolicDoublePendulumEnv(),
    SymbolicSwimmerEnv(),
    SymbolicCartpoleEnv(),
    SymbolicCartpoleSwingupEnv(),
]))

for env in envs:
    for init_controller_std in [0.1, 0.3, 0.5, 0.7, 1.0]:
        for seed in [11, 111, 211, 311, 411]:
            algo = WrappedILQR(
                env=env,
                n_itr=100,
                horizon=500,
                n_paths=100,
                init_controller_std=init_controller_std,
            )

            run_experiment_lite(
                algo.train(),
                n_parallel=4,
                seed=seed,
                mode="lab_kube",
                exp_prefix="gps_longer_horizon",
            )
