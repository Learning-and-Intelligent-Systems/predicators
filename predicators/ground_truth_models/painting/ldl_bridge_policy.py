"""Ground-truth LDL bridge policy for the painting environment."""

from pathlib import Path
from typing import Set

from predicators.ground_truth_models import GroundTruthLDLBridgePolicyFactory


class PaintingLDLBridgePolicyFactory(GroundTruthLDLBridgePolicyFactory):
    """Ground-truth LDL bridge policy for the painting environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"painting"}

    @classmethod
    def _get_ldl_file(cls) -> Path:
        return Path(__file__).parent / "painting_bridge_policy.ldl"
