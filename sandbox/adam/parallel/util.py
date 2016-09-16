

class SimpleContainer(object):
    """
    Container for convenient references.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
