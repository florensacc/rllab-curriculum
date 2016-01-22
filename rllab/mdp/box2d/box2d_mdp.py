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
    @autoargs.arg('position_only', type=bool,
                  help='Whether to only provide (generalized) position as the '
                       'observation (i.e. no velocities etc.)')
    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                       'problem non-Markovian!)')
    @autoargs.arg('action_noise', type=float,
                  help='Noise added to the controls, which will be '
                       'proportional to the action bounds')
    def __init__(
            self, model_path, frame_skip=1, position_only=False,
            obs_noise=0.0, action_noise=0.0):
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
        self.position_only = position_only
        self.obs_noise = obs_noise
        self.action_noise = action_noise
        self._action_bounds = None
        # cache the computation of position mask
        self._position_ids = None
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
        if self.position_only:
            return (len(self.get_position_ids()),)
        else:
            return (len(self.extra_data.states),)

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

    def compute_reward(self, action):
        """
        The implementation of this method should have two parts, structured
        like the following:

        <perform calculations before stepping the world>
        yield
        reward = <perform calculations after stepping the world>
        yield reward
        """
        raise NotImplementedError

    @overrides
    def step(self, state, action):
        """
        Note: override this method with great care, as it post-processes the
        observations, etc.
        """
        with self._set_state_tmp(state, restore=False):
            reward_computer = self.compute_reward(action)
            # forward the state
            next_state = state
            action = self.inject_action_noise(action)
            for _ in range(self.frame_skip):
                next_state = self.forward_dynamics(next_state, action,
                                                   restore=False)
            # notifies that we have stepped the world
            reward_computer.next()
            # actually get the reward
            reward = reward_computer.next()
            self._invalidate_state_caches()
            done = self.is_current_done()
            next_obs = self.get_current_obs()
            return next_state, next_obs, reward, done

    def filter_position(self, obs):
        """
        Filter the observation to contain only position information.
        """
        return obs[self.get_position_ids()]

    def get_obs_noise_scale_factor(self, obs):
        return np.ones_like(obs)

    def inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * \
            np.random.normal(size=obs.shape)
        return obs + noise

    def get_current_reward(
            self, state, xml_obs, action, next_state, next_xml_obs):
        raise NotImplementedError

    def is_current_done(self):
        raise NotImplementedError

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
            np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def get_current_obs(self):
        """
        This method should not be overwritten.
        """
        raw_obs = self.get_raw_obs()
        noisy_obs = self.inject_obs_noise(raw_obs)
        if self.position_only:
            return self.filter_position(noisy_obs)
        return noisy_obs

    def get_position_ids(self):
        if self._position_ids is None:
            self._position_ids = []
            for idx, state in enumerate(self.extra_data.states):
                if state.typ in ["xpos", "ypos", "apos"]:
                    self._position_ids.append(idx)
        return self._position_ids

    def get_raw_obs(self):
        """
        Return the unfiltered & noiseless observation. By default, it computes
        based on the declarations in the xml file.
        """
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
            elif state.joint:
                joint = find_joint(self.world, state.joint)
                if state.typ == "apos":
                    new_obs = joint.angle
                elif state.typ == "avel":
                    new_obs = joint.speed
                else:
                    raise NotImplementedError
            elif state.com:
                com_quant = self.compute_com_pos_vel(*state.com)
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

    def compute_com_pos_vel(self, *com):
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

    def get_com_position(self, *com):
        return self.compute_com_pos_vel(*com)[:2]

    def get_com_velocity(self, *com):
        return self.compute_com_pos_vel(*com)[2:]

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
