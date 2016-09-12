class Shareable(object):

    def __init__(self):
        self.shared_params = None

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

    def prepare_sharing(self):
        self.shared_params = self.extract_shared_params()
        self.set_shared_params(self.shared_params)

    def process_copy(self):
        """
        Copy itself (including all relevant params), but let the shareable params point to the shared memory.
        """
        assert(self.shared_params is not None)
        raise NotImplementedError
