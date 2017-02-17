import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.parameterized import Parameterized


class KerasPowered(Parameterized):
    def __init__(self, models):
        self._models = models
        Parameterized.__init__(self)

    def get_params_internal(self, **tags):
        assert 'regularizable' not in tags
        trainable = tags.get('trainable', None)
        if trainable is None:
            return self.weights
        elif trainable is True:
            return self.trainable_weights
        else:
            return self.non_trainable_weights

    @property
    def weights(self):
        return L.unique(sum([model.weights for model in self._models], []))

    @property
    def trainable_weights(self):
        return L.unique(sum([model.trainable_weights for model in self._models], []))

    @property
    def non_trainable_weights(self):
        return L.unique(sum([model.non_trainable_weights for model in self._models], []))

    @property
    def constraints(self):
        return {k: v for model in self._models for k, v in model.constraints.items()}
