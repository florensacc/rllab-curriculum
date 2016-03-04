from rl.envs.atari import AtariEnv
from rllab.mdp.base import MDP
from rllab.misc import autoargs
from rllab.misc.special import from_onehot
from rllab.core.serializable import Serializable
import theano

# pylint: disable=no-member
floatX = theano.config.floatX
# pylint: enable=no-member


class AtariMDP(MDP, Serializable):

    @autoargs.arg("rom_name", type=str,
                  help="Name of the game ROM, without the file type extension")
    @autoargs.arg("obs_type", type=str, choices=["ram", "image"],
                  help="Observation type. Must be either ram or image.")
    @autoargs.arg("frame_skip", type=int,
                  help="How many frames to skip between each step.")
    def __init__(self, rom_name="pong", obs_type="ram", frame_skip=4):
        Serializable.quick_init(self, locals())
        self._mdp = AtariEnv()

    def reset(self):
        raw_obs = self._mdp.reset()
        return self._normalize_obs(raw_obs)

    def step(self, action):
        a = from_onehot(action)
        ret = self._mdp.step(a)
        return self._normalize_obs(ret.observation), ret.reward, ret.done

    def _normalize_obs(self, obs):
        return (obs / 255.0) * 2 - 1.0

    @property
    def action_dim(self):
        return self._mdp.action_space.n

    @property
    def action_dtype(self):
        return 'uint8'

    @property
    def observation_dtype(self):
        return floatX

    @property
    def observation_shape(self):
        return self._mdp.observation_space.shape

    def plot(self):
        self._mdp.render()
