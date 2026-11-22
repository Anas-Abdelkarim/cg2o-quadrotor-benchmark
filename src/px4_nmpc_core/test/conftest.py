import pathlib
import sys

# allow `pytest core/px4_nmpc_core/test` from the source tree (without colcon install)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
