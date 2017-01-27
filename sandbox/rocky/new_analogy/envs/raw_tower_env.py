from sandbox.rocky.new_analogy.pymj import MjEnv

from rllab import config
import os


class RawTowerEnv(MjEnv):
    def __init__(self):

        xml_path = os.path.join(
            config.PROJECT_PATH,
            "sandbox/rocky/new_analogy/envs/raw_tower.xml"
        )
        xml = open(xml_path).read()

        # MjEnv.__init__(self, xml=xml, frame_skip=)
        pass

    pass
