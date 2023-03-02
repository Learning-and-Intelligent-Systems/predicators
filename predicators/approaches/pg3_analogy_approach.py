"""Policy-guided planning for generalized policy generation (PG3) initialized
with policies that are induced by analogy to another domain."""
from __future__ import annotations

import abc
import os
from typing import List, Set

import dill as pkl

from predicators import utils
from predicators.approaches.pg3_approach import PG3Approach
from predicators.envs import get_or_create_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import NSRT, LiftedDecisionList


class PG3AnalogyApproach(PG3Approach):
    """Policy-guided planning for generalized policy generation (PG3)
    initialized with policies that are induced by analogy to another domain."""

    @abc.abstractmethod
    def _induce_policies_by_analogy(
            self, base_policy: LiftedDecisionList, base_env: BaseEnv,
            target_env: BaseEnv, base_nsrts: Set[NSRT],
            target_nsrts: Set[NSRT]) -> List[LiftedDecisionList]:
        raise NotImplementedError("Override me!")

    def _get_policy_search_initial_ldls(self) -> List[LiftedDecisionList]:
        # Create base and target envs.
        base_env_name = CFG.pg3_init_base_env
        target_env_name = CFG.env
        base_env = get_or_create_env(base_env_name)
        target_env = get_or_create_env(target_env_name)
        base_options = get_gt_options(base_env.get_name())
        target_options = get_gt_options(target_env.get_name())
        base_nsrts = get_gt_nsrts(base_env.get_name(), base_env.predicates,
                                  base_options)
        target_nsrts = get_gt_nsrts(target_env.get_name(),
                                    target_env.predicates, target_options)
        # Initialize with initialized policy from file.
        if CFG.pg3_init_policy is None:  # pragma: no cover
            # By default, use policy from base domain.
            save_path = utils.get_approach_save_path_str()
            pg3_init_policy_file = f"{save_path}_None.ldl"
        else:
            pg3_init_policy_file = CFG.pg3_init_policy
        # Can load from a pickled LDL or a plain text LDL.
        _, file_extension = os.path.splitext(pg3_init_policy_file)
        assert file_extension in (".ldl", ".txt")
        if file_extension == ".ldl":
            with open(pg3_init_policy_file, "rb") as fb:
                base_policy = pkl.load(fb)
        else:
            with open(pg3_init_policy_file, "r", encoding="utf-8") as f:
                base_policy_str = f.read()
            base_policy = utils.parse_ldl_from_str(base_policy_str,
                                                   base_env.types,
                                                   base_env.predicates,
                                                   base_nsrts)
        target_policies = self._induce_policies_by_analogy(
            base_policy, base_env, target_env, base_nsrts, target_nsrts)
        return target_policies
