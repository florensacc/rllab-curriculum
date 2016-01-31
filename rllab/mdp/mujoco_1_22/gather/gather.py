from rlfun import Environment, Serializable, Discrete, Box
import numpy as np
import numpy.random as nr
from numpy import pi

RED = 0
GREEN = 1

colortups = {RED: (0, 0, 255), GREEN: (0, 255, 0)}


class Apple(object):

    def __init__(self, colorid, xy):
        self.colorid = colorid
        self.xy = xy

    @property
    def color(self):
        return colortups[self.colorid]


def random_apples(size):
    n_apples = int(size**2 / (pi * APPLE_RADIUS**2) / 15)
    apples = []
    for _ in xrange(n_apples):
        colorid = RED if np.random.rand() < .5 else GREEN
        x, y = np.random.randint(low=0, high=size, size=2)
        xy = (int(x), int(y))
        apples.append(Apple(colorid, xy))
    return apples


class GatherState(object):

    def __init__(self, xytheta, apples):
        self.xytheta = xytheta
        self.apples = apples

STOP = 0
FORWARD = 1
LEFT_TURN = 2
RIGHT_TURN = 3

SELF_COLOR = (255, 0, 0)
SELF_RADIUS = 7
APPLE_RADIUS = 7
MAX_DTHETA = .1
VEL = 1.0
SENSOR_RANGE = 60
SENSOR_RES = 11
SENSOR_FOV = pi / 2


def compute_vel_dtheta(a):
    if a == STOP:
        vel = 0
        dtheta = 0
    elif a == FORWARD:
        vel = VEL
        dtheta = 0
    elif a == LEFT_TURN:
        vel = VEL
        dtheta = - MAX_DTHETA
    elif a == RIGHT_TURN:
        vel = VEL
        dtheta = MAX_DTHETA
    return vel, dtheta

ZERO_MOTION = (0, 0, 0)


def compute_motion(xytold, xyt):
    xold, yold, told = xytold
    x, y, t = xyt
    dx = x - xold
    dy = y - yold
    dxl = np.cos(t) * dx + np.sin(t) * dy
    dyl = -np.sin(t) * dx + np.cos(t) * dy
    return (dxl, dyl, t - told)


class Gather(Environment, Serializable):
    """
    Gather apples

    Green apples are worth +1 points
    Red apples are worth -1 points
    """

    def __init__(self, size=200):
        Serializable.__init__(self, size)
        self.size = size
        self._last_scan = None

    def reset(self):
        x, y, theta = nr.rand(3)  # pylint: disable=W0612
        x = SELF_RADIUS + x * (self.size - SELF_RADIUS * 2)
        y = SELF_RADIUS + y * (self.size - SELF_RADIUS * 2)
        theta *= 2 * pi
        self.state = GatherState((x, y, theta), random_apples(self.size))
        obs, _rew = self._post_motion_step()
        return obs

    def step(self, a):

        vel, dtheta = compute_vel_dtheta(a)

        xo, yo, thetao = self.state.xytheta
        x = xo + vel * np.cos(thetao)
        y = yo + vel * np.sin(thetao)
        x = np.clip(x, SELF_RADIUS, self.size - SELF_RADIUS)
        y = np.clip(y, SELF_RADIUS, self.size - SELF_RADIUS)
        theta = thetao + dtheta
        self.state.xytheta = (x, y, theta)

        obs, reward = self._post_motion_step()

        return obs, reward, len(self.state.apples) == 0, {}

    def action_space(self):
        return Discrete(4)

    def observation_space(self):
        high = np.ones(SENSOR_RES * 3)
        low = -high
        return Box(low, high)

    def plot(self, wait=True):
        import cv2
        img = self._make_img()
        img = cv2.resize(
            img, (img.shape[1] * 2, img.shape[0] * 2))  # pylint: disable=E1101
        cv2.imshow("gather", img)  # pylint: disable=E1101
        scan = self._last_scan
        if scan is not None:
            scan = (scan * 255).reshape(1, 11, 3).astype('uint8')
            scan = cv2.resize(scan, (scan.shape[
                              1] * 20, scan.shape[0] * 20), interpolation=cv2.INTER_NEAREST)  # pylint: disable=E1101
            cv2.imshow('scan', scan)  # pylint: disable=E1101

        if wait:
            cv2.waitKey(20)  # pylint: disable=E1101

    def _post_motion_step(self):
        x, y, theta = self.state.xytheta
        scan = np.zeros((1, SENSOR_RES, 3), 'uint8')
        reward = 0

        newapples = []
        for apple in self.state.apples:
            ax, ay = apple.xy
            dist2 = (ax - x)**2 + (ay - y)**2
            dist = np.sqrt(dist2)
            if dist2 > SELF_RADIUS**2:
                newapples.append(apple)
            else:
                if apple.colorid == RED:
                    reward += 1
                else:
                    reward -= 1

            if dist2 < (APPLE_RADIUS + SENSOR_RANGE)**2:
                dy = ay - y
                dx = ax - x
                ang2center = np.arctan2(dy, dx)
                if np.cos(ang2center - theta) < 0:
                    continue
                for (i, dang) in enumerate(np.linspace(-SENSOR_FOV / 2, SENSOR_FOV / 2, SENSOR_RES)):
                    rayang = theta + dang
                    triang = rayang - ang2center
                    b = -2 * dist * np.cos(triang)
                    c = dist2 - APPLE_RADIUS**2
                    discrim = b**2 - 4 * c
                    if discrim > 0:
                        hitdist = (-b - np.sqrt(discrim)) / 2.0
                        intensity = np.clip(
                            int(255.0 * (SENSOR_RANGE - hitdist) / (SENSOR_RANGE - SELF_RADIUS)), 0, 255)
                        if apple.colorid == RED:
                            scan[0, i, 2] = intensity
                        elif apple.colorid == GREEN:
                            scan[0, i, 1] = intensity
                        else:
                            raise NotImplementedError
                    else:
                        pass

        self.state.apples = newapples
        scan = scan / 255.0
        scan = scan.reshape(scan.shape[0], -1)

        self._last_scan = scan

        return scan, reward

    def _make_img(self):
        import cv2
        state = self.state
        img = 255 + np.zeros((self.size, self.size, 3), 'uint8')
        for apple in self.state.apples:
            cv2.circle(img, apple.xy, APPLE_RADIUS,
                       apple.color, -1)  # pylint: disable=E1101
        x, y, theta = state.xytheta
        cv2.circle(img, (int(x), int(y)), SELF_RADIUS,
                   SELF_COLOR, -1)  # pylint: disable=E0602,E1101
        start = (int(x), int(y))
        for ang in np.linspace(theta - SENSOR_FOV / 2.0, theta + SENSOR_FOV / 2.0, SENSOR_RES):
            end = (int(x + SENSOR_RANGE * np.cos(ang)),
                   int(y + SENSOR_RANGE * np.sin(ang)))
            cv2.line(img, start, end, (0, 0, 0),
                     thickness=1)  # pylint: disable=E1101
        end = (int(x + SELF_RADIUS * np.cos(theta)),
               int(y + SELF_RADIUS * np.sin(theta)))
        cv2.line(img, start, end, (0, 255, 255),
                 thickness=2)  # pylint: disable=E1101

        return img
