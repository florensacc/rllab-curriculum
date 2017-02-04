import itertools
from copy import copy

from gym.monitoring.video_recorder import ImageEncoder
from gym.spaces import prng

from gpr.runner import BaseExperimentRunner
from rllab.misc.console import mkdir_p
from sandbox.rocky.new_analogy import fetch_utils
from sandbox.rocky.s3 import resource_manager
import tensorflow as tf
import cloudpickle as pickle
import joblib
from rllab import config
import os.path
import numpy as np

# resource_manager.register_file("fetch_5_blocks_dagger.pkl", "/tmp/params.pkl")

file_name = resource_manager.get_file("fetch_5_blocks_dagger.pkl")

print(file_name)


video_path = os.path.join(config.PROJECT_PATH, "data/video/fetch_5_blocks_dagger")
mkdir_p(video_path)

horizon = 2000


class RecordingRunner(BaseExperimentRunner):
    def __init__(self, env, policy, record_video=True):
        BaseExperimentRunner.__init__(self, TEST=False)
        self.env = fetch_utils.get_gpr_env(env)
        self.policy = policy
        self.x = None
        self.refreshed = False
        self.record_video = record_video

    def refresh(self, seed):#, env=None, world_params_update=None):
        '''
        If environment uses a camera than second viewer cannot be started.
        Here we prevent _get_obs() from being called (which would open an
        extra viewer).
        '''
        # if env is not None:
        #     self.env = env
        # if world_params_update is not None:
        #     self.world_params = self.world_params._replace(**world_params_update)
        # self.env.world_builder.world_params = self.world_params
        # self.env._world = None # forces to rebuild XML.
        self.env.seed(seed)
        prng.seed(seed)
        self.ob = self.env.reset()
        self.x = self.env.x
        world_filename = self.env.world.to_file()
        if not self.refreshed:
            self.viewer.set_model(world_filename)
            self.viewer.set_x(self.x, self.action)
            self.refreshed = True

    def run(self):
        for video_idx in itertools.count():
            self.refresh(seed=video_idx)
            cur_video_path = os.path.join(video_path, "video_{0:04d}.mp4".format(video_idx))
            cur_traj_path = os.path.join(video_path, "traj_{0:04d}.pkl".format(video_idx))
            width, height = self.viewer.get_dimensions()
            width -= width % 2
            height -= height % 2
            if self.record_video:
                self.viewer.image_encoder = ImageEncoder(cur_video_path, (height, width, 3), 60)
                self.viewer.record_video = True

            ob_n = []
            x_n = []
            a_n = []
            r_n = []

            for _ in range(horizon):
                ob_n.append(copy(self.ob))
                x_n.append(copy(self.x))
                self.action = self.policy.get_action(self.ob)[0]
                a_n.append(copy(self.action))
                self.ob, reward, done, _ = self.env.step(self.action)
                self.x = self.env.x
                self.reward = self.get_reward_info()
                r_n.append(copy(self.reward))
                self.set_state(self.env.x, self.action)
                self.delay_viewer_loop()
                if done:
                    break
            if self.record_video:
                self.viewer.image_encoder.close()
                self.viewer.record_video = False
                joblib.dump(dict(observations=np.asarray(ob_n), xs=np.asarray(x_n), actions=np.asarray(a_n),
                                 rewards=np.asarray(r_n)), cur_traj_path)


with tf.Session() as sess:
    data = joblib.load(file_name)

    policy = data['policy']
    env = data['env']

    policy = fetch_utils.DeterministicPolicy(env.spec, policy)

    RecordingRunner(env=env, policy=policy, record_video=False).run()
    #
    # import ipdb; ipdb.set_trace()
    #
    #
    # BaseExperimentRunner
