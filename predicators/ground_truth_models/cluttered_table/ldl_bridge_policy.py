"""Ground-truth LDL bridge policy for the cluttered table environment."""

from pathlib import Path
from typing import Set

from predicators.ground_truth_models import GroundTruthLDLBridgePolicyFactory


class ClutteredTableLDLBridgePolicyFactory(GroundTruthLDLBridgePolicyFactory):
    """Ground-truth LDL bridge policy for the cluttered table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cluttered_table"}

    @classmethod
    def _get_ldl_file(cls) -> Path:
        return Path(__file__).parent / "cluttered_table_bridge_policy.ldl"
