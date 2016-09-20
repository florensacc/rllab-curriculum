from sandbox.rocky.tf.envs.base import TfEnv


def unwrap(env):
    if isinstance(env, TfEnv):
        return unwrap(env.wrapped_env)
    return env
