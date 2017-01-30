from rllab.core.serializable import Serializable

class Kernel(Serializable):
    def get_kappa(self,xs):
        """
        Outputs a tensor (matrix) \kappa(x_j,x_k)
        indices: (j,k)
        """
        raise NotImplementedError

    def get_kappa_grads(self,xs):
        """
        Outputs a 3-D tensor \nabla_{x_j} \kappa(x_j,x_k)
        indices: (j,k,grads)
        """
        raise NotImplementedError

    def update(self, algo, feed_dict):
        """
        Adapt the kernel to data before computing SVGD
        """
        return {}
