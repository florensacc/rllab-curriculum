import gym
import numpy as np
from gym.spaces import Box
from gym.utils import seeding

from . import glfw
from .cymj import MjSim, MjViewerContext, Constants, MjParallelLite


class MjEnv(gym.Env):
    def __init__(self, xml, frame_skip, forward_after_steps):
        self.sim = MjSim(xml)
        self.model, self.data = self.sim.model, self.sim.data
        self.frame_skip = frame_skip
        self.forward_after_steps = forward_after_steps
        self.viewer = None
        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._seed()

        self.initialize()
        # observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        # assert not done
        observation = self._reset()
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low, high)

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def initialize(self):
        pass

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def viewer_prerender(self):
        """
        Called before each viewer render
        Useful for adding geoms to the viewer for visualization.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.sim.reset_data()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[...] = qpos
        self.data.qvel[...] = qvel
        # self.model._compute_subtree() #pylint: disable=W0212 # XXX implement this?
        self.sim.forward()

    @property
    def x(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    @x.setter
    def x(self, x):
        assert x.ndim == 1
        qpos, qvel = np.split(x, [self.model.nq])
        self.set_state(qpos, qvel)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl):
        self.data.ctrl[...] = ctrl
        self.sim.step(self.frame_skip)
        if self.forward_after_steps:
            self.sim.forward()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer_prerender()
        if mode == 'human':
            self.viewer.render()
        else:
            raise NotImplementedError(mode)

    def gen_contacting_bodies(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            yield tuple(sorted((
                self.sim.body_names[self.model.geom_bodyid[c.geom1]],
                self.sim.body_names[self.model.geom_bodyid[c.geom2]],
            )))


class MjViewer(object):
    GEOM_TYPE_MAP = {
        'sphere': Constants.GEOM_SPHERE,
        'capsule': Constants.GEOM_CAPSULE,
        'ellipsoid': Constants.GEOM_ELLIPSOID,
        'cylinder': Constants.GEOM_CYLINDER,
        'box': Constants.GEOM_BOX,
        'arrow': Constants.GEOM_ARROW,
        'arrow1': Constants.GEOM_ARROW1,
        'arrow2': Constants.GEOM_ARROW2,
    }

    def __init__(self, sim):
        self.sim = sim

        self.showinfo = True
        self.showhelp = False
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        self.lastbutton = 0
        self.lastclicktm = 0

        self.closed = False
        self.frame_count = 0
        self.geoms_to_add = []

        if not glfw.init():
            raise RuntimeError('glfwInit')
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(1200, 900, "Simulate", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not make window')

        # make context current, request v-sync on swapbuffers
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.ctx = MjViewerContext(self.sim)

        # set GLFW callbacks
        glfw.set_key_callback(self.window, self._keyboard)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_scroll_callback(self.window, self._scroll)
        glfw.set_window_refresh_callback(self.window, self._render)

        # center and scale view, update scene
        self.autoscale()
        self.ctx.update_scene()

        self.closed = False
        self.start_time = glfw.get_time()

    def close(self):
        if self.closed: return
        self.ctx.close()
        glfw.terminate()
        self.closed = True

    def render(self):
        if self.closed:
            return
        self._render(self.window)
        glfw.poll_events()  # calls all callbacks

    def overlay_tuples(self, gridpos, tuples, fontsize=0):
        """More common use case, display rows of tupled text"""
        if self.closed:
            return
        self.overlay(gridpos, [str(x[0]) for x in tuples], [str(x[1]) for x in tuples], fontsize=fontsize)

    def overlay(self, gridpos, title, content, fontsize=Constants.FONT_NORMAL):
        """Wrapper for mjr_overlay of text"""
        if self.closed:
            return
        if type(title) == list:
            title = '\n'.join(title)
        if type(content) == list:
            content = '\n'.join(content)
        width, height = glfw.get_framebuffer_size(self.window)
        self.ctx.overlay(fontsize, gridpos, width, height, title, content)

    def add_geom(self, geom_type, size, pos, mat, rgba, label=''):
        if geom_type not in self.GEOM_TYPE_MAP:
            raise RuntimeError('Unknown geom type: {}'.format(geom_type))
        self.geoms_to_add.append((self.GEOM_TYPE_MAP[geom_type], size, pos, mat, rgba, label.encode()))

    def autoscale(self):
        self.ctx.cam.lookat[0] = self.sim.model.stat.center[0]
        self.ctx.cam.lookat[1] = self.sim.model.stat.center[1]
        self.ctx.cam.lookat[2] = self.sim.model.stat.center[2]
        self.ctx.cam.distance = 1.5 * self.sim.model.stat.extent
        self.ctx.cam.type = Constants.CAMERA_FREE

    # ===== Callbacks =====

    def _keyboard(self, _window, key, scancode, act, mods):
        if self.closed:
            return

        # do not act on release
        if act == glfw.RELEASE:
            return

        if key == glfw.KEY_F1:  # help
            self.showhelp = not self.showhelp

        elif key == glfw.KEY_TAB:  # info
            self.showinfo = not self.showinfo

        elif key == glfw.KEY_ESCAPE:  # free camera
            self.ctx.cam.type = Constants.CAMERA_FREE

        elif key == glfw.KEY_LEFT_BRACKET:  # previous fixed camera or free
            if self.sim.model.ncam and self.ctx.cam.type == Constants.CAMERA_FIXED:
                if self.ctx.cam.fixedcamid > 0:
                    self.ctx.cam.fixedcamid -= 1
                else:
                    self.ctx.cam.type = Constants.CAMERA_FREE

        elif key == glfw.KEY_RIGHT_BRACKET:  # next fixed camera
            if self.sim.model.ncam:
                if self.ctx.cam.type != Constants.CAMERA_FIXED:
                    self.ctx.cam.type = Constants.CAMERA_FIXED
                    self.ctx.cam.fixedcamid = 0
                elif self.ctx.cam.fixedcamid < self.sim.model.ncam - 1:
                    self.ctx.cam.fixedcamid += 1

        elif key == glfw.KEY_SEMICOLON:  # cycle over frame rendering modes
            self.ctx.vopt.frame = max(0, self.ctx.vopt.frame - 1)

        elif key == glfw.KEY_APOSTROPHE:  # cycle over frame rendering modes
            self.ctx.vopt.frame = min(Constants.NFRAME - 1, self.ctx.vopt.frame + 1)

        elif key == glfw.KEY_PERIOD:  # cycle over label rendering modes
            self.ctx.vopt.label = max(0, self.ctx.vopt.label - 1)

        elif key == glfw.KEY_SLASH:  # cycle over label rendering modes
            self.ctx.vopt.label = min(Constants.NLABEL - 1, self.ctx.vopt.label + 1)

        elif key == glfw.KEY_A and (mods & glfw.MOD_CONTROL):
            self.autoscale()

    def _mouse_move(self, _window, xpos, ypos):
        if self.closed:
            return
        # no buttons down: nothing to do
        if not self.button_left and not self.button_middle and not self.button_right:
            return

        # compute mouse displacement, save
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # get current window size
        width, height = glfw.get_framebuffer_size(self.window)

        # get shift key state
        mod_shift = (
            glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(self.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # determine action based on mouse button
        if self.button_right:
            action = Constants.MOUSE_MOVE_H if mod_shift else Constants.MOUSE_MOVE_V
        elif self.button_left:
            action = Constants.MOUSE_ROTATE_H if mod_shift else Constants.MOUSE_ROTATE_V
        else:
            action = Constants.MOUSE_ZOOM

        # move camera
        self.ctx.move_camera(action, dx / height, dy / height)

    def _mouse_button(self, _window, button, act, mods):
        if self.closed:
            return

        # update button state
        self.button_left = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        # update mouse position
        self.lastx, self.lasty = glfw.get_cursor_pos(self.window)

        # save info
        if act == glfw.PRESS:
            self.lastbutton = button
            self.lastclicktm = glfw.get_time()

    def _scroll(self, _window, xoffset, yoffset):
        if self.closed:
            return
        # scroll: emulate vertical mouse motion = 5% of window height
        self.ctx.move_camera(Constants.MOUSE_ZOOM, 0, -0.05 * yoffset)

    def _render(self, _window):
        if self.closed:
            return

        self.frame_count += 1
        self.ctx.update_scene()

        for geom_type, size, pos, mat, rgba, label in self.geoms_to_add:
            self.ctx.add_geom(
                geom_type, size.astype(np.float32), pos.astype(np.float32),
                mat.astype(np.float32), rgba.astype(np.float32), label)
        del self.geoms_to_add[:]

        self.ctx.render(*glfw.get_framebuffer_size(self.window))

        # Help overlays
        if self.showhelp:
            help_menu = [
                ('Help', 'F1'),
                ('Tab', 'Info'),
                ('Ctrl-A', 'Autoscale'),
                ('ESC', 'Free Camera'),
                ('[ ]', 'Camera'),
                ("; '", 'Frame'),
                ('. /', 'Label'),
            ]
            self.overlay_tuples(Constants.GRID_TOPLEFT, help_menu)

        # Info overlays
        if self.showinfo:
            convergence = np.log10(max(self.sim.convergence, 1e-10))
            if convergence > -3:
                quality = "poor"
            elif convergence > -6:
                quality = "moderate"
            else:
                quality = "very good"
            if self.ctx.cam.type == Constants.CAMERA_FREE:
                camstr = "Free"
            elif self.ctx.cam.type == Constants.CAMERA_TRACKING:
                camstr = "Tracking"
            else:
                camstr = "Fixed %d" % self.ctx.cam.fixedcamid
            info_menu = [
                ("FPS", "%.1f" % (self.frame_count / (glfw.get_time() - self.start_time))),
                ("Convergence", "%.1f (%s)" % (convergence, quality)),
                ("Nr. iterations", "%d" % self.sim.data.solver_iter),
                ("Camera", camstr),
                ("#Cameras", self.sim.model.ncam),
                ("Frame", Constants.frame_string(self.ctx.vopt.frame).decode()),
                ("Label", Constants.label_string(self.ctx.vopt.label).decode()),
            ]
            self.overlay_tuples(Constants.GRID_BOTTOMLEFT, info_menu)

        glfw.swap_buffers(self.window)
