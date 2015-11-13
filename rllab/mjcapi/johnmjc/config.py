import tensorfuse as theano
import os.path as osp
from path import Path

floatX = theano.config.floatX

PROJECT_ROOT = Path(__file__) / ".." / ".."/ ".."
CTRL_ROOT = PROJECT_ROOT / "vendor" / "john_control"
MJC_DATA_DIR = PROJECT_ROOT / "src" / "mjc" / "mjcdata"
