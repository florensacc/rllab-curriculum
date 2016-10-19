from cython.parallel import parallel, prange
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
import cython
cimport numpy as np
from libc.string cimport memcpy
from libcpp cimport bool


cdef extern from "ale_interface.hpp":
    enum Action:
        PLAYER_A_NOOP = 0
        PLAYER_A_FIRE = 1
        PLAYER_A_UP = 2
        PLAYER_A_RIGHT = 3
        PLAYER_A_LEFT = 4
        PLAYER_A_DOWN = 5
        PLAYER_A_UPRIGHT = 6
        PLAYER_A_UPLEFT = 7
        PLAYER_A_DOWNRIGHT = 8
        PLAYER_A_DOWNLEFT = 9
        PLAYER_A_UPFIRE = 10
        PLAYER_A_RIGHTFIRE = 11
        PLAYER_A_LEFTFIRE = 12
        PLAYER_A_DOWNFIRE = 13
        PLAYER_A_UPRIGHTFIRE = 14
        PLAYER_A_UPLEFTFIRE = 15
        PLAYER_A_DOWNRIGHTFIRE = 16
        PLAYER_A_DOWNLEFTFIRE = 17
        PLAYER_B_NOOP = 18
        PLAYER_B_FIRE = 19
        PLAYER_B_UP = 20
        PLAYER_B_RIGHT = 21
        PLAYER_B_LEFT = 22
        PLAYER_B_DOWN = 23
        PLAYER_B_UPRIGHT = 24
        PLAYER_B_UPLEFT = 25
        PLAYER_B_DOWNRIGHT = 26
        PLAYER_B_DOWNLEFT = 27
        PLAYER_B_UPFIRE = 28
        PLAYER_B_RIGHTFIRE = 29
        PLAYER_B_LEFTFIRE = 30
        PLAYER_B_DOWNFIRE = 31
        PLAYER_B_UPRIGHTFIRE = 32
        PLAYER_B_UPLEFTFIRE = 33
        PLAYER_B_DOWNRIGHTFIRE = 34
        PLAYER_B_DOWNLEFTFIRE = 35
        RESET = 40
        UNDEFINED = 41
        RANDOM = 42
        SAVE_STATE = 43
        LOAD_STATE = 44
        SYSTEM_RESET = 45
        LAST_ACTION_INDEX = 50

    cppclass ALEScreen:
        const size_t height() nogil
        const size_t width() nogil

        const np.uint8_t*getArray() nogil

    cppclass ALERAM:
        const size_t size() nogil

        const np.uint8_t*array() nogil

    cppclass ALEInterface:
        ALEInterface() nogil

        int getInt(const string& key) nogil

        bool getBool(const string& key) nogil

        float getFloat(const string& key) nogil

        void setString(const string& key, const string& value) nogil

        void setInt(const string& key, const int value) nogil

        void setBool(const string& key, const bool value) nogil

        void setFloat(const string& key, const float value) nogil

        void loadROM(string rom_file) nogil

        int act(Action action) nogil

        int lives() nogil

        bool game_over() nogil

        void reset_game() nogil

        vector[Action] getLegalActionSet() nogil

        vector[Action] getMinimalActionSet() nogil

        const ALEScreen& getScreen() nogil

        const ALERAM& getRAM() nogil


cdef class ParAtari(object):
    cdef int n_envs
    cdef vector[ALEInterface*] games
    # cdef vector[int] start_lives
    cdef string rom_path

    def __cinit__(self, int n_envs, const string& rom_path):
        self.n_envs = n_envs
        self.rom_path = rom_path
        self.games.resize(n_envs)
        # self.start_lives.resize(n_envs)
        self.create_all()

    @cython.boundscheck(False)
    def create_all(self, np.ndarray[np.uint8_t, cast=True] mask=None):
        cdef int i
        cdef bool no_mask = mask is None
        for i in range(self.n_envs):
            if no_mask or mask[i]:
                if self.games[i] != NULL:
                    del self.games[i]
                self.games[i] = new ALEInterface()
                self.games[i].loadROM(self.rom_path)
                # self.start_lives[i] = self.games[i].lives()

    @cython.boundscheck(False)
    def reset_game_all(self, np.ndarray[np.uint8_t, cast=True] mask=None):
        cdef int i
        cdef bool no_mask = mask is None
        with nogil, parallel():
            for i in prange(self.n_envs, schedule='static'):
                if no_mask or mask[i]:
                    self.games[i].reset_game()
                    # self.start_lives[i] = self.games[i].lives()

    @cython.boundscheck(False)
    def load_rom(self, int i, const string& rom_file):
        self.games[i].loadROM(rom_file)

    @cython.boundscheck(False)
    def set_float(self, int i, const string& key, float value):
        self.games[i].setFloat(key, value)

    @cython.boundscheck(False)
    def set_bool(self, int i, const string& key, bool value):
        self.games[i].setBool(key, value)

    @cython.boundscheck(False)
    def set_int(self, int i, const string& key, int value):
        self.games[i].setInt(key, value)

    @cython.boundscheck(False)
    def get_screen_width(self, int i):
        return self.games[i].getScreen().width()

    @cython.boundscheck(False)
    def get_screen_height(self, int i):
        return self.games[i].getScreen().height()

    @cython.boundscheck(False)
    def get_ram_size(self, int i):
        return self.games[i].getRAM().size()

    @cython.boundscheck(False)
    def get_minimal_action_set(self, int i):
        return self.games[i].getMinimalActionSet()

    @cython.boundscheck(False)
    def act_all(
            self,
            int frame_skip,
            np.ndarray[int, ndim=1, mode="c"] actions,
            np.ndarray[int, ndim=1, mode="c"] out_rewards,
    ):
        cdef int i
        cdef int i_step
        with nogil, parallel():
            for i in prange(self.n_envs, schedule='static'):
                out_rewards[i] = 0
                for i_step in range(frame_skip):
                    if self.games[i].game_over():
                        break
                    out_rewards[i] += self.games[i].act(<Action> actions[i])

    @cython.boundscheck(False)
    def get_game_screen_all(self, np.ndarray[np.uint8_t, ndim=2, mode="c"] output_buffer not None,
                            np.ndarray[np.uint8_t, cast=True] mask=None):
        cdef np.uint8_t*screen
        cdef int i
        cdef int width
        cdef int height
        cdef bool no_mask = mask is None
        cdef int idx = 0

        for i in range(self.n_envs):
            if no_mask or mask[i]:
                width = self.games[i].getScreen().width()
                height = self.games[i].getScreen().height()
                screen = self.games[i].getScreen().getArray()
                memcpy(&output_buffer[idx, 0], screen, width * height)
                idx += 1

    @cython.boundscheck(False)
    def get_ram_all(self, np.ndarray[np.uint8_t, ndim=2, mode="c"] output_buffer not None):
        cdef np.uint8_t*ram
        cdef int i
        cdef int width
        cdef int height

        for i in range(self.n_envs):
            size = self.games[i].getRAM().size()
            ram = self.games[i].getRAM().array()
            memcpy(&output_buffer[i, 0], ram, size)

    @cython.boundscheck(False)
    def get_game_over_all(self, np.ndarray[np.uint8_t, ndim=1, mode="c"] output_buffer not None):
        for i in range(self.n_envs):
            output_buffer[i] = self.games[i].game_over()

    @cython.boundscheck(False)
    def get_lives_all(self, np.ndarray[int, ndim=1, mode="c"] output_buffer not None):
        for i in range(self.n_envs):
            output_buffer[i] = self.games[i].lives()
