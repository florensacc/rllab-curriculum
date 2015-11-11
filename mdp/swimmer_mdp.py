"""multi-link swimmer moving in a fluid."""

import numpy as np
from .base import SymbolicMDP
import tensorfuse as theano
import tensorfuse.tensor as TT
#from theano.tensor.slinalg import solve
from misc.overrides import overrides
from misc.viewer2d import Viewer2D, Colors
from misc.ext import extract


__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"

cnt = 0


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    yout = [None] * len(t)
    yout[0] = y0
    i = 0
    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = derivs(y0, thist, *args, **kwargs)
        k2 = derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs)
        k3 = derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs)
        k4 = derivs(y0 + dt * k3, thist + dt, *args, **kwargs)
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


class SwimmerMDP(SymbolicMDP):

    """
    A swimmer consisting of a chain of d links connected by rotational joints.
    Each joint is actuated. The goal is to move the swimmer to a specified goal
    position.

    *States*:
        | 2 dimensions: position of nose relative to goal
        | d -1 dimensions: angles
        | 2 dimensions: velocity of the nose
        | d dimensions: angular velocities

    *Actions*:
        each joint torque is discretized in 3 values: -2, 0, 2

    .. note::
        adapted from Yuval Tassas swimmer implementation in Matlab available at
        http://www.cs.washington.edu/people/postdocs/tassa/code/

    .. seealso::
        Tassa, Y., Erez, T., & Smart, B. (2007).
        *Receding Horizon Differential Dynamic Programming.*
        In Advances in Neural Information Processing Systems.
    """
    dt = 0.05
    episodeCap = 1000
    discount_factor = 0.98

    def __init__(self, d=3, k1=7.5, k2=0.3, horizon=400):
        """
        d:
            number of joints
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.nose = 0
        masses = np.ones(d)
        lengths = np.ones(d)
        inertia = masses * lengths * lengths / 12.
        goal = np.zeros(2)

        # reward function parameters
        self.cu = 0.04
        self.cx = 2.

        Q = np.eye(self.d, k=1) - np.eye(self.d)
        Q[-1, :] = masses
        A = np.eye(self.d, k=1) + np.eye(self.d)
        A[-1, -1] = 0.
        P = np.dot(np.linalg.inv(Q), A * lengths[None, :]) / 2.

        U = np.eye(self.d) - np.eye(self.d, k=-1)
        U = U[:, :-1]

        G = np.dot(P.T * masses[None, :], P)

        floatX =theano.config.floatX

        self.masses = masses.astype(floatX)
        self.lengths = lengths.astype(floatX)
        self.inertia = inertia.astype(floatX)
        self.goal = goal.astype(floatX)
        self.P = P.astype(floatX)
        self.U = U.astype(floatX)
        self.G = G.astype(floatX)

        self._n_actions = d - 1
        self._state_shape = (2*d+4,)
        self._observation_shape = (2*d+3,)

        self._viewer = None

        super(SwimmerMDP, self).__init__(horizon)


    @property
    def state_bounds(self):
        d = self.d
        lb = np.ones((2*d+4,)) * -np.inf
        ub = np.ones((2*d+4,)) * np.inf
        return lb, ub

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def state_shape(self):
        return self._state_shape

    def reset_sym(self):
        theta = TT.zeros(self.d)
        pos_cm = TT.constant([10., 0.])
        v_cm = TT.zeros(2)
        dtheta = TT.zeros(self.d)
        state = TT.concatenate([pos_cm, theta, v_cm, dtheta]) 
        obs = self.observation_sym(state)
        return state, obs

    @overrides
    def observation_sym(self, state):
        return TT.concatenate(self.body_coord_symbolic(state))

    @overrides
    def forward_sym(self, state, action):
        d = self.d
        action = TT.clip(action*2, -2., 2.)
        ns = rk4( dsdt, state, [0, self.dt], action, self.P, self.inertia, self.G, self.U, self.lengths, self.masses, self.k1, self.k2, d)[-1] 
        return ns

    @overrides
    def reward_sym(self, state, action):
        xrel = self.body_coord_symbolic(state)[0] - self.goal
        dist = TT.sum(xrel ** 2)
        return (
            - self.cx * dist / (TT.sqrt(dist) + 1) - self.cu * TT.sum(action ** 2)
        )

    @overrides
    def cost_sym(self, state, action):
        return - self.reward_sym(state, action)

    @overrides
    def final_cost_sym(self, state):
        return TT.constant(0)

    @overrides
    def done_sym(self, state):
        return TT.constant(False)

    def get_joint_coords(self, state):
        theta, pos_cm = extract(
            decode_state(state, self.d),
            "theta", "pos_cm"
        )
        T = np.empty((self.d, 2))
        T[:, 0] = np.cos(theta)
        T[:, 1] = np.sin(theta)
        R = np.dot(self.P, T)
        R1 = R - .5 * self.lengths[:, None] * T
        R2 = R + .5 * self.lengths[:, None] * T
        Rx = np.hstack([R1[:, 0], [R2[-1, 0]]]) + pos_cm[0]
        Ry = np.hstack([R1[:, 1], [R2[-1, 1]]]) + pos_cm[1]
        return zip(Rx, Ry)

    # calculate the center of mass for the given state
    #def calc_com(self, state):
    #    points = np.array(self.get_joint_coords(state))
    #    weight_sum = np.zeros_like(points[0])
    #    total_mass = 0
    #    for idx, pts in enumerate(zip(points, points[1:])):
    #        p1, p2 = pts
    #        weight_sum += self.masses[idx] * (p1 + p2) / 2
    #        total_mass += self.masses[idx]
    #    return weight_sum / total_mass

    def plot(self, states=None, actions=None):
        global cnt
        cnt += 1
        if self._viewer is None:
            self._viewer = Viewer2D(xlim=[5, 15], ylim=[-5, 5])

        if states is None:
            states = [self.state]
        if actions is None and self.action is not None:
            actions = [self.action]

        # center the viewer around the center of mass of the last state
        #if cnt > 50:
        #    import ipdb; ipdb.set_trace()
        center = decode_state(states[-1], self.d)["pos_cm"][:2]

        viewer = self._viewer
        viewer.reset()
        viewer.xlim = (center[0] - 2.5, center[0] + 2.5)
        viewer.ylim = (center[1] - 2.5, center[1] + 2.5)
        viewer.checker(offset=center)

        d = self.d
        for state in states:
            points = self.get_joint_coords(state)
            for p1, p2 in zip(points, points[1:]):
                viewer.line(p1, p2, Colors.blue, width=0.1)
            for p in points:
                viewer.circle(p, radius=0.07, color=Colors.red)
        viewer.loop_once()

    def body_coord_symbolic(self, state):
        """
        transforms the current state into coordinates that are more
        reasonable for learning
        returns a 4-tupel consisting of:
        nose position, joint angles (d-1), nose velocity, angular velocities

        The nose position and nose velocities are referenced to the nose rotation.
        """
        d = self.d
        pos_cm, theta, v_cm, dtheta = extract(
            decode_state(state, self.d),
            "pos_cm", "theta", "v_cm", "dtheta"
        )

        cth = TT.cos(theta)
        sth = TT.sin(theta)
        M = self.P - 0.5 * TT.diag(self.lengths)
        #  stores the vector from the center of mass to the nose
        c2n = TT.stack([TT.dot(M[self.nose], cth), TT.dot(M[self.nose], sth)])
        #  absolute position of nose
        T = -pos_cm - c2n - self.goal
        #  rotating coordinate such that nose is axis-aligned (nose frame)
        #  (no effect when  \theta_{nose} = 0)
        c2n_x = TT.stack([cth[self.nose], sth[self.nose]])
        c2n_y = TT.stack([-sth[self.nose], cth[self.nose]])
        Tcn = TT.stack([TT.sum(T * c2n_x), TT.sum(T * c2n_y)])

        #  velocity at each joint relative to center of mass velocity
        vx = -TT.dot(M, sth * dtheta)
        vy = TT.dot(M, cth * dtheta)
        #  velocity at nose (world frame) relative to center of mass velocity
        v2n = TT.stack([vx[self.nose], vy[self.nose]])
        #  rotating nose velocity to be in nose frame
        Vcn = TT.stack([TT.sum((v_cm + v2n) * c2n_x),
                        TT.sum((v_cm + v2n) * c2n_y)])
        #  angles should be in [-pi, pi]
        ang = TT.mod(
            theta[1:] - theta[:-1] + np.pi,
            2 * np.pi) - np.pi
        return Tcn, ang, Vcn, dtheta


def decode_state(state, d):
    pos_cm = state[:2]
    theta = state[2:2+d]
    v_cm = state[2+d:4+d]
    dtheta = state[4+d:]
    return dict(pos_cm=pos_cm, theta=theta, v_cm=v_cm, dtheta=dtheta)

def dsdt(s, t, a, P, I, G, U, lengths, masses, k1, k2, d):
    """
    time derivative of system dynamics
    """

    theta, v_cm, dtheta = extract(
        decode_state(s, d),
        "theta", "v_cm", "dtheta"
    )

    cth = TT.cos(theta)
    sth = TT.sin(theta)
    rVx = TT.dot(P, -sth * dtheta)
    rVy = TT.dot(P, cth * dtheta)
    Vx = rVx + v_cm[0]
    Vy = rVy + v_cm[1]

    Vn = -sth * Vx + cth * Vy
    Vt = cth * Vx + sth * Vy

    EL1 = TT.dot((v1Mv2(-sth, G, cth) + v1Mv2(cth, G, sth)) * dtheta[None, :]
                 + (v1Mv2(cth, G, -sth) + v1Mv2(sth, G, cth)) * dtheta[:, None], dtheta)
    EL3 = TT.diag(I) + v1Mv2(sth, G, sth) + v1Mv2(cth, G, cth)
    EL2 = - k1 * TT.dot((v1Mv2(-sth, P.T, -sth) + v1Mv2(cth, P.T, cth)) * lengths[None, :], Vn) \
          - k1 * TT.power(lengths, 3) * dtheta / 12. \
          - k2 * \
        TT.dot((v1Mv2(-sth, P.T, cth) + v1Mv2(cth, P.T, sth))
               * lengths[None, :], Vt)

    return TT.concatenate([
        v_cm,
        dtheta,
        TT.reshape(-(k1 * TT.sum(-sth * Vn) + k2 * TT.sum(cth * Vt)) / TT.sum(masses), (1,)),
        TT.reshape(-(k1 * TT.sum(cth * Vn) + k2 * TT.sum(sth * Vt)) / TT.sum(masses), (1,)),
        TT.dot(TT.nlinalg.matrix_inverse(EL3), EL1 + EL2 + TT.dot(U, a))
    ])

def v1Mv2(v1, M, v2):
    """
    computes diag(v1) dot M dot diag(v2).
    returns np.ndarray with same dimensions as M
    """
    return v1[:, None] * M * v2[None, :]
