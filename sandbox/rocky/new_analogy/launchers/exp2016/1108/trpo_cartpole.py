from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.new_analogy.tf.algos.pposgd_joint_ac import PPOSGD
from sandbox.rocky.new_analogy.tf.policies.gaussian_rnn_actor_critic import GaussianRNNActorCritic
from sandbox.rocky.tf.envs.base import TfEnv


def run_task(*_):

    env = TfEnv(normalize(CartpoleEnv()))

    ac = GaussianRNNActorCritic(name="ac", env_spec=env.spec)

    algo = PPOSGD(
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        env=env,
        policy=ac,
        baseline=ac,
        optimizer=TBPTTOptimizer(n_epochs=5, batch_size=40),
    )

    algo.train()

run_task()
