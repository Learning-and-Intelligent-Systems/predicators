"""This implementation is heavily based on the pybullet-planning repository by
Caelan Garrett (https://github.com/caelan/pybullet-planning/)."""
from typing import List, NamedTuple


class IKFastInfo(NamedTuple):
    module_dir: str
    module_name: str
    base_link: str
    ee_link: str
    free_joints: List[str]
