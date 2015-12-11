import theano
import os.path as osp
from path import Path

floatX = theano.config.floatX

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CTRL_ROOT = str(PROJECT_ROOT / "vendor" / "john_mjc")
MJC_DATA_DIR = str(PROJECT_ROOT / "src" / "mjc" / "mjcdata")
