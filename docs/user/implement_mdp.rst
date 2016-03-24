.. _implement_mdp:

=====================
Implementing New MDPs
=====================

In this section, we will walk through an example of implementing a point robot
MDP using our framework.

Each MDP should implement at least the following methods / properties defined
in the file :code:`rllab/mdp/base.py`:

.. code-block:: py

    class MDP(object):

        @property
        def observation_shape(self):
            raise NotImplementedError

        @property
        def action_dim(self):
            raise NotImplementedError

        @property
        def observation_dtype(self):
            raise NotImplementedError

        @property
        def action_dtype(self):
            raise NotImplementedError

        @property
        def action_bounds(self):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def reset(self):
            raise NotImplementedError

We will implement a simple MDP with 2D observations and 2D actions. The goal is
to control a point robot in 2D to move it to the origin. We receive position of
a point robot in the 2D plane :math:`(x, y) \in \mathbb{R}^2`. The action is
its velocity :math:`(\dot x, \dot y) \in \mathbb{R}^2` constrained so that
:math:`|\dot x| \leq 0.1` and :math:`|\dot y| \leq 0.1`. We encourage the robot
to move to the origin by defining its reward as the negative distance to the
origin: :math:`r(x, y) = - \sqrt{x^2 + y^2}`.

We start by creating a new file for the MDP. We assume that it is placed under
:code:`examples/point_mdp.py`. First, let's declare a class inheriting from
the base MDP:

.. code-block:: py

    from rllab.mdp.base import MDP

    class PointMDP(MDP):

        # ...

For each MDP, we will need to specify the shape of its observations, and the
dimensionality of its actions. This is done by implementing the following
property methods:

.. code-block:: py

    class PointMDP(MDP):

        # ...

        @property
        def observation_shape(self):
            return (2,)

        @property
        def action_dim(self):
            return 2

In addition, we also need to specify the data types for the observations and
the actions. This should be something that Numpy / Theano accepts for the
:code:`dtype` argument. Since we typically use `Theano <http://deeplearning.net/software/theano/>`_
for implementing the algorithms, we make it so that the data type conform to Theano's
configuration:

.. code-block:: py

    import theano

    class PointMDP(MDP):

        # ...

        @property
        def observation_dtype(self):
            return theano.config.floatX

        @property
        def action_dtype(self):
            return theano.config.floatX

We should also specify the bounds for the action. This is done by returning a
tuple of lower bounds and upper bounds for each action dimension.

.. code-block:: py

    import numpy as np

    class PointMDP(MDP):

        # ...

        @property
        def action_bounds(self):
            return - 0.1 * np.ones(2), 0.1 * np.ones(2)

Now onto the interesting part, where we actually implement the dynamics for the
MDP. This is done through two methods, :code:`reset` and
:code:`step`. The :code:`reset` method randomly initializes the state
of the MDP according to some initial state distribution. To keep things simple,
we will just sample the coordinates from a uniform distribution. The method
should also return the corresponding observation. In our case, it is just the
same as its state.

.. code-block:: py

    class PointMDP(MDP):

        # ...

        def reset(self):
            self._state = np.random.uniform(-1, 1, size=(2,))
            observation = np.copy(self._state)
            return observation

The :code:`step` method takes an action and advances the state of the
MDP. It should return a tuple, containing the next observation, the reward, and
a flag indicating whether the episode is terminated after taking the step (in
which case, the next observation will be ignored). The procedure that
interfaces with the MDP is responsible for calling :code:`reset` after seeing
that the episode is terminated.

.. code-block:: py

    class PointMDP(MDP):

        # ...

        def step(self, action):
            self._state = self._state + action
            x, y = self._state
            reward = - (x**2 + y**2) ** 0.5
            done = abs(x) < 0.01 and abs(y) < 0.01
            next_observation = np.copy(self._state)
            return next_observation, reward, done

Finally, we can implement some plotting to visualize what the MDP is doing. For
simplicity, let's just print the current state of the MDP on the terminal:

.. code-block:: py

    class PointMDP(MDP):

        # ...

        def plot(self):
            print 'current state:', self._state

And we're done! We can now simulate the mdp using the following diagnostic
script:

.. code-block:: bash

    python scripts/sim_mdp.py --mdp examples.point_mdp --mode random

It simulates an episode of the MDP with random actions, sampled from a uniform
distribution within the defined action bounds.

You could also train a neural network policy to solve the task:

.. code-block:: bash

    python scripts/run_experiment.py --mdp examples.point_mdp --algo trpo --policy mean_std_nn_policy --baseline linear_feature_baseline
