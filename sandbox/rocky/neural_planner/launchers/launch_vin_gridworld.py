from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.neural_planner.tmux_utils import TmuxSession
import time

tsess = TmuxSession("D0", logdir=None)

for gpu_id in range(8):
    tsess.new_window("gpu%d" % gpu_id)
    tsess.send_keys("gpu%d" % gpu_id, "source ~/.bashrc")
    tsess.send_keys(
        "gpu%d" % gpu_id,
        """
        THEANO_FLAGS="device=cpu" CUDA_VISIBLE_DEVICES="%d" python sandbox/rocky/neural_planner/launchers/vin_gridworld.py
        """
        % gpu_id)
    time.sleep(2)


tsess.new_window("htop")
tsess.send_keys("htop", "htop")
