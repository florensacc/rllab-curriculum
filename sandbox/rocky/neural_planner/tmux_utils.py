from __future__ import print_function
from __future__ import absolute_import

import os


def prep_tmux(logdir=None):
    if logdir is not None:
        mkdir = ["mkdir -p {}".format(logdir)]
    else:
        mkdir = []
    cmds = (
        mkdir +
        [
            "cd " + os.path.dirname(__file__),
            "tmux kill-session",
        ])
    for cmd in cmds:
        os.system(cmd)


def new_session(sess_name, job_name):
    os.system("tmux new-session -s {} -n {} -d".format(sess_name, job_name))


def new_window(sess_name, job_name):
    os.system("tmux new-window -t {} -n {}".format(sess_name, job_name))


def send_keys(cmd, job_name):
    os.system("tmux send-keys -t {} '{}' Enter".format(job_name, cmd))


class TmuxSession(object):
    def __init__(self, session_name, logdir):

        prep_tmux(logdir)

        self._session_name = session_name
        self._uninited = True
        self._windows = dict()
        self._id_to_window = dict()
        self._num_windows = 0

    def new_window(self, window_name):
        assert window_name not in self._windows
        if self._uninited:
            new_session(self._session_name, window_name)
        else:
            new_window(self._session_name, window_name)
        self._windows[window_name] = 1
        self._uninited = False

    def send_keys(self, window_name, cmd):
        print("sending ({}): {}".format(window_name, cmd))
        send_keys(cmd, window_name)

    def send_keys_to_all(self, cmd):
        for window_name in self._windows:
            self.send_keys(window_name, cmd)
