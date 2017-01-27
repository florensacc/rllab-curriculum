# !/usr/bin/env python3
import click
import logging
import sys
import os
import six.moves.cPickle as pickle
import gym
from gym.monitoring.video_recorder import ImageEncoder

gym.undo_logger_setup()
import numpy as np

np.set_printoptions(precision=3)
logger = logging.getLogger('bin/view_trajectory')
import glob


@click.command()
@click.argument('fname')
@click.option('--verbose', '-v',
              help="Sets the debug noise level, specify multiple times "
                   "for more verbosity.",
              type=click.IntRange(0, 3, clamp=True),
              count=True)
def main(fname, verbose):
    logger_handler = logging.StreamHandler(sys.stderr)
    logger_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(logger_handler)
    logging.getLogger().setLevel(DEBUG_LOGGING_MAP.get(verbose, logging.DEBUG))

    pkl = glob.glob(os.path.expanduser(fname))[0]
    traj = pickle.load(open(pkl, "rb"))

    encoder = ImageEncoder(
        output_path="video.mp4",
        frame_shape=(1000, 1000, 3),
        frames_per_sec=100  # 4
    )
    env = traj.env
    for t, x in enumerate(traj.solution["x"]):
        print(t)
        env.reset_to(x)
        img = env.render(mode='rgb_array')
        encoder.capture_frame(img)
    encoder.close()


DEBUG_LOGGING_MAP = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}

if __name__ == '__main__':
    main()
