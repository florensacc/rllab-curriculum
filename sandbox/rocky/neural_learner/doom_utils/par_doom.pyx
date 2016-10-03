from cython.parallel import parallel, prange
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np
from libcpp cimport bool


cdef extern from "ViZDoom.h" namespace "vizdoom":
    enum ScreenResolution:
        pass

    enum ScreenFormat:
        CRCGCB = 0  # 3 channels of 8-bit values in RGB order
        CRCGCBDB = 1  # 4 channels of 8-bit values in RGB + depth buffer order
        RGB24 = 2  # channel of RGB values stored in 24 bits where R value is stored in the oldest 8 bits
        RGBA32 = 3  # channel of RGBA values stored in 32 bits where R value is stored in the oldest 8 bits
        ARGB32 = 4  # channel of ARGB values stored in 32 bits where A value is stored in the oldest 8 bits
        CBCGCR = 5  # 3 channels of 8-bit values in BGR order
        CBCGCRDB = 6  # 4 channels of 8-bit values in BGR + depth buffer order
        BGR24 = 7  # channel of BGR values stored in 24 bits where B value is stored in the oldest 8 bits
        BGRA32 = 8  # channel of BGRA values stored in 32 bits where B value is stored in the oldest 8 bits
        ABGR32 = 9  # channel of ABGR values stored in 32 bits where A value is stored in the oldest 8 bits
        GRAY8 = 10  # 8-bit gray channel
        DEPTH_BUFFER8 = 11  # 8-bit depth buffer channel
        DOOM_256_COLORS8 = 12

    enum Button:
        pass

    enum Mode:
        pass

    cppclass DoomGame:
        DoomGame() nogil

        void close() nogil

        void setViZDoomPath(string path) nogil

        void setDoomGamePath(string path) nogil

        void setDoomScenarioPath(string path) nogil

        void setDoomMap(string path) nogil

        void setScreenResolution(ScreenResolution resolution) nogil

        void setScreenFormat(ScreenFormat format) nogil

        void setRenderHud(bool hud) nogil

        void setRenderCrosshair(bool crosshair) nogil

        void setRenderWeapon(bool weapon) nogil

        void setRenderDecals(bool decals) nogil

        void setRenderParticles(bool particles) nogil

        void setLivingReward(double livingReward) nogil

        void setWindowVisible(bool visibility) nogil

        void setSoundEnabled(bool sound) nogil

        void addAvailableButton(Button button) nogil

        void setMode(Mode mode) nogil

        void newEpisode() nogil

        bool init() nogil

        ScreenFormat getScreenFormat() nogil

        int getScreenWidth() nogil

        int getScreenHeight() nogil

        int getScreenChannels() nogil

        const np.uint8_t* getGameScreen() nogil

        void setAction(const vector[int]& actions) nogil

        void advanceAction(unsigned int tics, bool updateState, bool renderOnly) nogil

        bool isEpisodeFinished() nogil

        double getTotalReward() nogil



cdef class ParDoom(object):
    cdef int n_envs
    cdef vector[DoomGame*] games

    def __cinit__(self, int n_envs):
        self.n_envs = n_envs
        self.games.resize(n_envs)
        self.create_all()

    def close_all(self, np.uint8_t[:] mask=None):
        cdef int i
        cdef bool no_mask = mask is None
        with nogil, parallel():
            for i in prange(self.n_envs):
                if no_mask or mask[i]:
                    if self.games[i] != NULL:
                        self.games[i].close()
                        del self.games[i]
                        self.games[i] = NULL

    def create_all(self, np.uint8_t[:] mask=None):
        cdef int i
        cdef bool no_mask = mask is None
        with nogil, parallel():
            for i in prange(self.n_envs):
                if no_mask or mask[i]:
                    if self.games[i] != NULL:
                        self.games[i].close()
                        del self.games[i]
                    self.games[i] = new DoomGame()

    def init_all(self, np.uint8_t[:] mask=None):
        cdef int i
        cdef bool no_mask = mask is None
        with nogil, parallel():
            for i in prange(self.n_envs):
                if no_mask or mask[i]:
                    self.games[i].init()

    def new_episode_all(self, np.uint8_t[:] mask=None):
        cdef int i
        cdef bool no_mask = mask is None
        with nogil, parallel():
            for i in prange(self.n_envs):
                if no_mask or mask[i]:
                    self.games[i].newEpisode()

    def set_vizdoom_path(self, int i, const string& path):
        self.games[i].setViZDoomPath(path)

    def set_doom_game_path(self, int i, const string& path):
        self.games[i].setDoomGamePath(path)

    def set_doom_scenario_path(self, int i, const string& path):
        self.games[i].setDoomScenarioPath(path)

    def set_doom_map(self, int i, const string& map):
        self.games[i].setDoomMap(map)

    def set_screen_resolution(self, int i, ScreenResolution resolution):
        self.games[i].setScreenResolution(resolution)

    def set_screen_format(self, int i, ScreenFormat format):
        self.games[i].setScreenFormat(format)

    def set_render_hud(self, int i, bool hud):
        self.games[i].setRenderHud(hud)

    def set_render_crosshair(self, int i, bool crosshair):
        self.games[i].setRenderCrosshair(crosshair)

    def set_render_weapon(self, int i, bool weapon):
        self.games[i].setRenderWeapon(weapon)

    def set_render_decals(self, int i, bool decals):
        self.games[i].setRenderDecals(decals)

    def set_render_particles(self, int i, bool particles):
        self.games[i].setRenderParticles(particles)

    def set_living_reward(self, int i, double livingReward):
        self.games[i].setLivingReward(livingReward)

    def set_window_visible(self, int i, bool visible):
        self.games[i].setWindowVisible(visible)

    def set_sound_enabled(self, int i, bool sound):
        self.games[i].setSoundEnabled(sound)

    def add_available_button(self, int i, Button button):
        self.games[i].addAvailableButton(button)

    def set_mode(self, int i, Mode mode):
        self.games[i].setMode(mode)

    def set_action(self, int i, np.ndarray[int, ndim=1, mode="c"] actions not None):
        cdef vector[int] vec_actions
        vec_actions.assign(&actions[0], &actions[-1]+1)
        self.games[i].setAction(vec_actions)

    def get_game_screen_shape(self, int i):
        cdef ScreenFormat format = self.games[i].getScreenFormat()
        cdef int channels = self.games[i].getScreenChannels()
        cdef int width = self.games[i].getScreenWidth()
        cdef int height = self.games[i].getScreenHeight()

        if format == ScreenFormat.CRCGCB or \
                        format == ScreenFormat.CRCGCBDB or \
                        format == ScreenFormat.CBCGCR or \
                        format == ScreenFormat.CBCGCRDB or \
                        format == ScreenFormat.GRAY8 or \
                        format == ScreenFormat.DEPTH_BUFFER8 or \
                        format == ScreenFormat.DOOM_256_COLORS8:
            return (channels, height, width)
        else:
            return (height, width, channels)

    def get_game_screen_all(self):
        ret = []
        cdef np.uint8_t[:,:,:] screen
        for i in range(self.n_envs):
            shape = self.get_game_screen_shape(i)
            screen = <np.uint8_t[:shape[0],:shape[1],:shape[2]]> self.games[i].getGameScreen()
            ret.append(np.asarray(screen))#screen[:shape[0], :shape[1], :shape[2]], dtype=np.uint8))
        return ret

    def advance_action_all(self, unsigned int tics, bool update_state, bool render_only):
        cdef int i
        with nogil, parallel():
            for i in prange(self.n_envs):
                self.games[i].advanceAction(tics, update_state, render_only)

    def is_episode_finished(self, int i):
        return self.games[i].isEpisodeFinished()

    def get_total_reward(self, int i):
        return self.games[i].getTotalReward()
