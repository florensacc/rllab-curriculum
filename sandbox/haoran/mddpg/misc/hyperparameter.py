import abc
import copy
import math
import random


class Hyperparameter(object):
    def __init__(self, name):
        self._name = name
        self._last_value = None

    @property
    def name(self):
        return self._name

    @abc.abstractclassmethod
    def generate_next_value(self):
        """Return a value for the hyperparameter"""
        return

    def generate(self):
        self._last_value = self.generate_next_value()
        return self._last_value


class EnumParam(Hyperparameter):
    def __init__(self, name, possible_values):
        super(LogFloatParam, self).__init__(name)
        self.possible_values = possible_values

    def generate_next_value(self):
        return random.choice(self.possible_values)


class LogFloatParam(Hyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LogFloatParam, self).__init__(name)
        self._linear_float_param = LinearFloatParam("log_" + name,
                                                    math.log(min_value),
                                                    math.log(max_value))

    def generate_next_value(self):
        return math.e ** (self._linear_float_param.generate())


class LinearFloatParam(Hyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LinearFloatParam, self).__init__(name)
        self._min = min_value
        self._delta = max_value - min_value

    def generate_next_value(self):
        return random.random() * self._delta + self._min


class FixedParam(Hyperparameter):
    def __init__(self, name, value):
        super(FixedParam, self).__init__(name)
        self._value = value

    def generate_next_value(self):
        return self._value


class HyperparameterSweeper(object):
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters or []
        self.validate_hyperparameters()
        self.default_kwargs = {}

    def validate_hyperparameters(self):
        names = set()
        for hp in self.hyperparameters:
            name = hp.name
            if name in names:
                raise Exception("Hyperparameter '{0}' already added.".format(
                    name))
            names.add(name)

    def set_default_parameters(self, default_kwargs):
        self.default_kwargs = default_kwargs

    def generate_random_hyperparameters(self):
        kwargs = copy.deepcopy(self.default_kwargs)
        for hp in self.hyperparameters:
            kwargs[hp.name] = hp.generate()
        return kwargs

    def sweep_hyperparameters(self, function, num_configs):
        returned_value_and_params = []
        for _ in range(num_configs):
            kwargs = self.generate_random_hyperparameters()
            score = function(**kwargs)
            returned_value_and_params.append((score, kwargs))

        return returned_value_and_params
