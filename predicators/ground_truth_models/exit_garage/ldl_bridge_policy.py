"""Ground-truth LDL bridge policy for the exit garage environment."""

from pathlib import Path
from typing import Set

from predicators.ground_truth_models import GroundTruthLDLBridgePolicyFactory


class ExitGarageLDLBridgePolicyFactory(GroundTruthLDLBridgePolicyFactory):
    """Ground-truth LDL bridge policy for the exit garage environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"exit_garage"}

    @classmethod
    def _get_ldl_file(cls) -> Path:
        return Path(__file__).parent / "exit_garage_bridge_policy.ldl"
