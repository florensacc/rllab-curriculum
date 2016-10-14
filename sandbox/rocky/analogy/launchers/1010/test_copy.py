import joblib
import tensorflow as tf
import numpy as np

from rllab.misc import logger
from sandbox.rocky.analogy.algos.trainer import collect_demo
from sandbox.rocky.analogy.utils import unwrap

with tf.Session() as sess:
    params = joblib.load("/tmp/params.pkl")
    env = params['env']
    unwrap(env).reset_trial()
    policy = params['policy']
    trainer = params['trainer']

    print("Checking for uninitialized variables...")
    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    if len(uninitialized_vars) > 0:
        print("Initializing uninitialized variables")
        sess.run(tf.initialize_variables(uninitialized_vars))
    print("All variables initialized")

    data_dict = dict(
        demo_paths=[],
        analogy_paths=[],
        demo_seeds=[],
        analogy_seeds=[],
        target_seeds=[],
        analogy_envs=[],
    )

    for _ in range(50):
        demo_path, analogy_path, demo_seed, analogy_seed, target_seed = collect_demo(
            G=None,
            demo_collector=trainer.demo_collector,
            demo_seed=np.random.randint(low=0, high=1000000),
            analogy_seed=np.random.randint(low=0, high=1000000),
            target_seed=np.random.randint(low=0, high=1000000),
            env_cls=trainer.env_cls,
            horizon=trainer.horizon,
        )
        data_dict["demo_paths"].append(demo_path)
        data_dict["analogy_paths"].append(analogy_path)
        data_dict["demo_seeds"].append(demo_seed)
        data_dict["analogy_seeds"].append(analogy_seed)
        data_dict["target_seeds"].append(target_seed)
        data_dict["analogy_envs"].append(trainer.env_cls(seed=analogy_seed, target_seed=target_seed))

    trainer.eval_and_log(policy, data_dict)

    logger.dump_tabular()
