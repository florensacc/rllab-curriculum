from sandbox.pchen.dqn.dqn_ported import DQNP
from sandbox.pchen.dqn.envs.atari import AtariEnvCX

env = AtariEnvCX("pong", obs_type='image')
DQNP(env).train()