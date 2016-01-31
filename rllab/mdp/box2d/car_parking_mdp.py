import pygame

from rllab.mdp.box2d.box2d_mdp import Box2DMDP
from rllab.mdp.box2d.parser import find_body
import numpy as np
from rllab.core.serializable import Serializable
from rllab.mdp.box2d.parser.xml_box2d import _get_name
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class CarParkingMDP(Box2DMDP, Serializable):

    @autoargs.inherit(Box2DMDP.__init__)
    @autoargs.arg("random_start", type=bool,
                  help="Randomized starting position by uniforming sampling starting car angle"
                       "and position from a circle of radius 5")
    def __init__(self, *args, **kwargs):
        super(CarParkingMDP, self).__init__(
            self.model_path("car_parking.xml"),
            *args, **kwargs
        )
        self.random_start = kwargs.get("random_start", True)
        self.goal = find_body(self.world, "goal")
        self.car = find_body(self.world, "car")
        self.wheels = [body for body in self.world.bodies if "wheel" in _get_name(body)]
        self.front_wheels = [body for body in self.wheels if "front" in _get_name(body)]
        self.max_deg = 30.
        self.goal_radius = 1.
        self.vel_thres = 1e-1
        self.start_radius = 5.
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def before_world_step(self, state, action):
        desired_angle = self.car.angle + action[-1]/180*np.pi
        for wheel in self.front_wheels:
            wheel.angle = desired_angle
            wheel.angularVelocity = 0 # kill angular velocity

        # kill all wheels' lateral speed
        for wheel in self.wheels:
            ortho = wheel.GetWorldVector((1, 0))
            lateral_speed = wheel.linearVelocity.dot(ortho) * ortho
            impulse = wheel.mass * -lateral_speed
            wheel.ApplyLinearImpulse(impulse, wheel.worldCenter, True)
            # also apply a tiny bit of fraction
            mag = wheel.linearVelocity.dot(wheel.linearVelocity)
            if mag != 0:
                wheel.ApplyLinearImpulse(0.1 * wheel.mass * -wheel.linearVelocity/mag**0.5, wheel.worldCenter, True)


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
        if self.random_start:
            pos_angle, car_angle = np.random.rand(2) * np.pi * 2
            self.car.position = (self.start_radius*np.cos(pos_angle), self.start_radius*np.sin(pos_angle))
            self.car.angle = car_angle
        return self.get_state(), self.get_current_obs()

    @overrides
    def compute_reward(self, action):
        yield
        not_done = not self.is_current_done()
        dist_to_goal = self.get_current_obs()[-3]
        yield - 1*not_done - 2*dist_to_goal

    @overrides
    def is_current_done(self):
        pos_satified = np.linalg.norm(self.car.position) <= self.goal_radius
        vel_satisfied = np.linalg.norm(self.car.linearVelocity) <= self.vel_thres
        return pos_satified and vel_satisfied

    @overrides
    def action_from_keys(self, keys):
        go = np.zeros(self.action_dim)
        if keys[pygame.K_LEFT]:
            go[-1] = self.max_deg
        if keys[pygame.K_RIGHT]:
            go[-1] = -self.max_deg
        if keys[pygame.K_UP]:
            go[0] = 10
        if keys[pygame.K_DOWN]:
            go[0] = -10
        return go

