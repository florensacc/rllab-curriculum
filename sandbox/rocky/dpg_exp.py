from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.dpg.dpg import DPG
from sandbox.rocky.dpg.continuous_mlp_policy import ContinuousMLPPolicy
from sandbox.rocky.dpg.continuous_mlp_q_function import ContinuousMLPQFunction
from sandbox.rocky.dpg.ou_strategy import OUStrategy
import lasagne.nonlinearities as NL


stub(globals())


env = normalize(CartpoleEnv(), scale_reward=1)
algo = DPG(
    qf_learning_rate=1e-5,
    policy_learning_rate=1e-5,
    # policy_weight_decay=0.01,
    max_path_length=100,
)
policy = ContinuousMLPPolicy(env_spec=env.spec, hidden_sizes=(8,))#400,300))#, bn=True)#,
# hidden_nonlinearity=NL.tanh)#,
# bn=True)
es = OUStrategy(env_spec=env.spec, theta=0.15, sigma=0.2)
qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=(8,))#400,300))#, bn=True)#, hidden_nonlinearity=NL.tanh)#,
# bn=True)

run_experiment_lite(
    algo.train(env=env, policy=policy, qf=qf, es=es),
    n_parallel=4,
    snapshot_mode="last",
)
