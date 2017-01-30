class Shareable(object):
    def extract_shared_params(self):
        """
        Create a dictionary of shared RawArrays extracted from shareable params. The RawArrays are created by deep copies.
        """
        return dict()

    def set_shared_params(self, params):
        """
        Set the parameters from a dictionary of shared RawArrays.
        This gives a shallow copy.
        """
        pass

    def share_params(self):
        """
        Create shared deep copies of self's shareable params. Let self's params be shallow copies of them. Then return the shared params.
        """
        shared_params = self.extract_shared_params()
        self.set_shared_params(shared_params)
        return shared_params
