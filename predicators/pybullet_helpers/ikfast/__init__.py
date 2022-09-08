"""The `ikfast` module contains all the functionality to compile, install, and
load the IKFast module and also run IK using it."""

from typing import List, NamedTuple


class IKFastInfo(NamedTuple):
    """IKFast information for a given robot."""
    module_dir: str
    module_name: str
    base_link: str
    ee_link: str
    free_joints: List[str]
