from rllab.mdp.box2d.parser import world_from_xml
from rllab.mdp.box2d.parser.xml_box2d import ExtraData, XmlBox2D, _get_name
import argparse
import xml.etree.ElementTree as ET
from examples.framework import Framework
from rllab.misc.logger import record_tabular, dump_tabular
from rllab.misc.overrides import overrides

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the b2 xml file')

    args, _ = parser.parse_known_args()
    file = args.file
    class TestXml(Framework):
        name = file

        def __init__(self):
            super(TestXml, self).__init__()

            with open(file, "r") as f:
                s = f.read()
            extra_data = ExtraData()
            world_model = XmlBox2D.from_xml(ET.fromstring(s))
            world = world_model.to_box2d(extra_data, self.world)
            world, extra_data = world_from_xml(s)

        @overrides
        def Step(self, settings):
            for joint in self.world.joints:
                name = _get_name(joint)
                if name:
                    record_tabular("%s angle" % name, joint.angle)

            dump_tabular()
            super(TestXml, self).Step(settings)
    TestXml().run()

