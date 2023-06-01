"""Ground-truth LDL bridge policy for the stick button environment."""

from pathlib import Path
from typing import Set

from predicators.ground_truth_models import GroundTruthLDLBridgePolicyFactory


class StickButtonLDLBridgePolicyFactory(GroundTruthLDLBridgePolicyFactory):
    """Ground-truth LDL bridge policy for the stick button environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"stick_button"}

    @classmethod
    def _get_ldl_file(cls) -> Path:
        return Path(__file__).parent / "stick_button_bridge_policy.ldl"
