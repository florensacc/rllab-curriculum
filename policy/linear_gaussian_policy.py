from .base import Policy
from misc.overrides import overrides
import numpy as np

class LinearGaussianPolicy(Policy):

    def __init__(self, xref, uref, K, k, Quu):
        self.xref = xref
        self.uref = uref
        self.K = K
        self.k = k
        self.Quu = Quu

    def get_pdist(self, state, timestep):
        mean = self.uref[timestep] + self.k[timestep] + self.K[timestep].dot(state - self.xref[timestep])
        log_std = 0.5*np.log(np.diag(np.linalg.inv(self.Quu[timestep])))
        return mean, log_std
