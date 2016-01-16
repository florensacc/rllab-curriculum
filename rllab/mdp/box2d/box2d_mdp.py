from contextlib import contextmanager
import numpy as np
import os.path as osp
from rllab.mdp.box2d.parser.xml_box2d import world_from_xml, find_body, \
    find_joint
from rllab.mdp.box2d.box2d_viewer import Box2DViewer
from rllab.mdp.base import ControlMDP
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class Box2DMDP(ControlMDP):

    @autoargs.arg("frame_skip", type=int,
                  help="Number of frames to skip")
    def __init__(self, model_path, frame_skip=1):
        with open(model_path, "r") as f:
            s = f.read()
        world, extra_data = world_from_xml(s)
        self.world = world
        self.extra_data = extra_data
        self.initial_state = self.get_state()
        self.current_state = self.initial_state
        self.viewer = None
        self.frame_skip = frame_skip
        self.timestep = self.extra_data.timeStep
        self._action_bounds = None
        self._observation_shape = None
        self._cached_obs = None
        self._cached_coms = {}

    def model_path(self, file_name):
        return osp.abspath(osp.join(osp.dirname(__file__),
                                    'models/%s' % file_name))

    def _set_state(self, state):
        splitted = np.array(state).reshape((-1, 6))
        for body, body_state in zip(self.world.bodies, splitted):
            xpos, ypos, apos, xvel, yvel, avel = body_state
            body.position = (xpos, ypos)
            body.angle = apos
            body.linearVelocity = (xvel, yvel)
            body.angularVelocity = avel

    @property
    @overrides
    def state_shape(self):
        return (len(self.world.bodies) * 6,)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        return self.get_state(), self.get_current_obs()

    def _invalidate_state_caches(self):
        self._cached_obs = None
        self._cached_coms = {}

    def get_state(self):
        s = []
        for body in self.world.bodies:
            s.append(np.concatenate([
                list(body.position),
                [body.angle],
                list(body.linearVelocity),
                [body.angularVelocity]
            ]))
        return np.concatenate(s)

    @property
    @overrides
    def action_dim(self):
        return len(self.extra_data.controls)

    @property
    @overrides
    def action_dtype(self):
        return 'float32'

    @property
    @overrides
    def observation_dtype(self):
        return 'float32'

    @property
    @overrides
    def observation_shape(self):
        if not self._observation_shape:
            self._observation_shape = self.get_current_obs().shape
        return self._observation_shape

    @property
    @overrides
    def action_bounds(self):
        if not self._action_bounds:
            lb = [control.ctrllimit[0] for control in self.extra_data.controls]
            ub = [control.ctrllimit[1] for control in self.extra_data.controls]
            self._action_bounds = (np.array(lb), np.array(ub))
        return self._action_bounds

    @contextmanager
    def _set_state_tmp(self, state, restore=True):
        if np.array_equal(state, self.current_state) and not restore:
            yield
        else:
            prev_state = self.current_state
            self._set_state(state)
            yield
            if restore:
                self._set_state(prev_state)
            else:
                self.current_state = self.get_state()

    @overrides
    def forward_dynamics(self, state, action, restore=True):
        if len(action) != self.action_dim:
            raise ValueError('incorrect action dimension: expected %d but got '
                             '%d' % (self.action_dim, len(action)))
        with self._set_state_tmp(state, restore):
            lb, ub = self.action_bounds
            action = np.clip(action, lb, ub)
            for ctrl, act in zip(self.extra_data.controls, action):
                if ctrl.typ == "force":
                    assert ctrl.body
                    body = find_body(self.world, ctrl.body)
                    direction = np.array(ctrl.direction)
                    direction = direction / np.linalg.norm(direction)
                    world_force = body.GetWorldVector(direction * act)
                    world_point = body.GetWorldPoint(ctrl.anchor)
                    body.ApplyForce(world_force, world_point, wake=True)
                elif ctrl.typ == "torque":
                    assert ctrl.joint
                    joint = find_joint(self.world, ctrl.joint)
                    joint.motorEnabled = True
                    # forces the maximum allowed torque to be taken
                    if act > 0:
                        joint.motorSpeed = 1e5
                    else:
                        joint.motorSpeed = -1e5
                    joint.maxMotorTorque = abs(act)
                else:
                    raise NotImplementedError
            self.world.Step(
                self.extra_data.timeStep,
                self.extra_data.velocityIterations,
                self.extra_data.positionIterations
            )
            return self.get_state()

    @overrides
    def step(self, state, action):
        raw_obs = self.get_raw_obs()
        next_state = state
        for _ in range(self.frame_skip):
            next_state = self.forward_dynamics(next_state, action,
                                               restore=False)
        self._invalidate_state_caches()
        done = self.is_current_done()
        next_raw_obs = self.get_raw_obs()
        next_obs = self.get_current_obs()
        reward = self.get_current_reward(
            state, raw_obs, action, next_state, next_raw_obs)
        return next_state, next_obs, reward, done

    def get_current_reward(
            self, state, raw_obs, action, next_state, next_raw_obs):
        raise NotImplementedError

    def is_current_done(self):
        raise NotImplementedError

    def get_current_obs(self):
        return self.get_raw_obs()

    def get_raw_obs(self):
        if self._cached_obs is not None:
            return self._cached_obs
        obs = []
        for state in self.extra_data.states:
            new_obs = None
            if state.body:
                body = find_body(self.world, state.body)
                if state.local is not None:
                    l = state.local
                    position = body.GetWorldPoint(l)
                    linearVel = body.GetLinearVelocityFromLocalPoint(l)
                    # now I wish I could write angle = error "not supported"
                else:
                    position = body.position
                    linearVel = body.linearVelocity

                if state.typ == "xpos":
                    new_obs = position[0]
                elif state.typ == "ypos":
                    new_obs = position[1]
                elif state.typ == "xvel":
                    new_obs = linearVel[0]
                elif state.typ == "yvel":
                    new_obs = linearVel[1]
                elif state.typ == "apos":
                    new_obs = body.angle
                elif state.typ == "avel":
                    new_obs = body.angularVelocity
                else:
                    raise NotImplementedError
                obs.append(new_obs)
            elif state.joint:
                joint = find_joint(self.world, state.joint)
                if state.typ == "apos":
                    new_obs = joint.angle
                elif state.typ == "avel":
                    new_obs = joint.speed
                else:
                    raise NotImplementedError
            elif state.com:
                com_quant = self._compute_com(state.com)
                if state.typ == "xpos":
                    new_obs = com_quant[0]
                elif state.typ == "ypos":
                    new_obs = com_quant[1]
                elif state.typ == "xvel":
                    new_obs = com_quant[2]
                elif state.typ == "yvel":
                    new_obs = com_quant[3]
                else:
                    print state.typ
                    # orientation and angular velocity of the whole body is not
                    # supported
                    raise NotImplementedError
            else:
                raise NotImplementedError

            if state.transform is not None:
                if state.transform == "id":
                    pass
                elif state.transform == "sin":
                    new_obs = np.sin(new_obs)
                elif state.transform == "cos":
                    new_obs = np.cos(new_obs)
                else:
                    raise NotImplementedError

            obs.append(new_obs)

        self._cached_obs = np.array(obs)
        return self._cached_obs

    def _compute_com(self, com):
        com_key = ",".join(sorted(com))
        if com_key in self._cached_coms:
            return self._cached_coms[com_key]
        total_mass_quant = 0
        total_mass = 0
        for body_name in com:
            body = find_body(self.world, body_name)
            total_mass_quant += body.mass * np.array(
                list(body.worldCenter) + list(body.linearVelocity))
            total_mass += body.mass
        com_quant = total_mass_quant / total_mass
        self._cached_coms[com_key] = com_quant
        return com_quant

    @overrides
    def start_viewer(self):
        if not self.viewer:
            self.viewer = Box2DViewer(self.world)
        return self.viewer

    @overrides
    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()
        self.viewer = None

    @overrides
    def plot(self, states=None, actions=None, pause=False):
        if states or actions or pause:
            raise NotImplementedError
        if self.viewer:
            self.viewer.loop_once()
