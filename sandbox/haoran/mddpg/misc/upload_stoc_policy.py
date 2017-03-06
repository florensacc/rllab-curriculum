"""
Extract the low-level policy, save it alone as a .pkl file, and then
    upload it to s3.
"""
import os
import joblib
from sandbox.haoran.myscripts.retrainer import Retrainer
from sandbox.haoran.mddpg.policies.stochastic_policy_theano import \
    StochasticNNPolicy

CONVERT_TO_THEANO = True

exp_list = [
    dict(
        exp_prefix="tuomas/vddpg/exp-000b",
        exp_name="exp-000b_20170213_150257_941463_tuomas_ant",
        snapshot_file="itr_499.pkl",
    )
]
for exp in exp_list:
    retrainer = Retrainer(
        exp_prefix=exp["exp_prefix"],
        exp_name=exp["exp_name"],
        snapshot_file=exp["snapshot_file"],
        configure_script="",
    )
    env = retrainer.get_env()
    tf_policy = retrainer.get_policy()

    if CONVERT_TO_THEANO:
        file_name = exp["snapshot_file"].split('.pkl')[0] + "_theano_policy.pkl"
        policy = StochasticNNPolicy.copy_from_tf_policy(
            env_spec=env.spec,
            tf_policy=tf_policy,
        )
    else:
        file_name = exp["snapshot_file"].split('.pkl')[0] + "_tf_policy.pkl"
        policy = tf_policy

    local_file_name = os.path.join(
        retrainer.local_log_dir,
        file_name,
    )
    remote_file_name = os.path.join(
        retrainer.remote_log_dir,
        file_name,
    )
    joblib.dump(policy, local_file_name)
    os.system("""
        aws s3 cp {local} {remote}
    """.format(
        local=local_file_name,
        remote=remote_file_name,
    ))
