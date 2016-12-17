from rllab.core.serializable import Serializable

class BaseHallucinator(Serializable):
    """
    Hallucinate additional samples for the latents variables by naive ancestral sampling.
    """

    def __init__(self, env_spec, policy, n_hallucinate_samples=5):
        """
        :param policy:
        :param n_hallucinate_samples:
        :return:
        """
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy
        self.n_hallucinate_samples = n_hallucinate_samples

    def hallucinate(self, samples_data):  # here samples data have already been processed, so it has all Adv, Ret,..
        raise NotImplementedError
