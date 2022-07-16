from typing import List, NamedTuple


class IKFastInfo(NamedTuple):
    module_dir: str
    module_name: str
    base_link: str
    ee_link: str
    free_joints: List[str]
