from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.mdp.box2d.parser.xml_box2d import _get_name
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class CarParkingMDP(Box2DMDP, Serializable):

    @autoargs.inherit(Box2DMDP.__init__)
    def __init__(self, *args, **kwargs):
        super(CarParkingMDP, self).__init__(
            self.model_path("car_parking.xml"),
            *args, **kwargs
        )
        self.goal = find_body(self.world, "goal")
        self.car = find_body(self.world, "car")
        self.wheels = [body for body in self.world.bodies if "wheel" in _get_name(body)]
        self.max_deg = 30
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def before_world_step(self, state, action):
        # kill all wheels' lateral speed
        for wheel in self.wheels:
            ortho = wheel.GetWorldVector((1, 0))
            lateral_speed = wheel.linearVelocity.dot(ortho) * ortho
            impulse = wheel.mass * -lateral_speed
            wheel.ApplyLinearImpulse(impulse, wheel.worldCenter, True)
            # wheel.ApplyAngularImpulse(0.1*wheel.inertia*-wheel.angularVelocity)

    @property
    @overrides
    def action_dim(self):
        return super(CarParkingMDP, self).action_dim + 1

    @property
    @overrides
    def action_bounds(self):
        lb, ub = super(CarParkingMDP, self).action_bounds
        return np.append(lb, -self.max_deg), np.append(ub, self.max_deg)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        return self.get_state(), self.get_current_obs()

    @overrides
    def compute_reward(self, action):
        yield
        yield 0

    @overrides
    def is_current_done(self):
        return False

