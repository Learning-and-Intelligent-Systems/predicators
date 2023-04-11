"""Learn an LDL bridge policy from online demonstrations."""

import logging
from typing import Dict, List, Set, Tuple

from predicators import utils
from predicators.bridge_policies.ldl_bridge_policy import LDLBridgePolicy
from predicators.structs import NSRT, BridgeDataset, GroundAtom, LDLRule, \
    LiftedAtom, LiftedDecisionList, ParameterizedOption, Predicate, Type, \
    _GroundNSRT


class LearnedLDLBridgePolicy(LDLBridgePolicy):
    """A learned LDL bridge policy."""

    def __init__(self, types: Set[Type], predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        super().__init__(types, predicates, options, nsrts)
        self._current_ldl = LiftedDecisionList([])

    @classmethod
    def get_name(cls) -> str:
        return "learned_ldl"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_ldl(self) -> LiftedDecisionList:
        return self._current_ldl

    def learn_from_demos(self, dataset: BridgeDataset) -> None:
        # Convert dataset into (atoms, ground NSRT) dataset where atoms
        # includes failure atoms.
        ground_atom_data: List[Tuple[Set[GroundAtom], _GroundNSRT]] = []
        # Collect all seen atoms for later constructing negative preconds.
        all_seen_atoms: Set[GroundAtom] = set()
        for failed_option_set, ground_nsrt, ground_atoms, _ in dataset:
            # Add failure atoms.
            ground_atoms |= utils.get_failure_atoms(failed_option_set)
            all_seen_atoms.update(ground_atoms)
            ground_atom_data.append((ground_atoms, ground_nsrt))

        # Convert dataset into NSRT: [lifted atoms] dataset where the
        # atoms are only over the objects in the NSRT. Do this for both
        # positive and negative atoms.
        nsrt_to_pos_lifted_atoms: Dict[NSRT, List[Set[LiftedAtom]]] = {}
        nsrt_to_neg_lifted_atoms: Dict[NSRT, List[Set[LiftedAtom]]] = {}
        for ground_atoms, ground_nsrt in ground_atom_data:
            nsrt = ground_nsrt.parent
            sub = dict(zip(ground_nsrt.objects, nsrt.parameters))
            pos_lifted_atoms = {
                a.lift(sub)
                for a in ground_atoms if all(o in sub for o in a.objects)
            }
            if nsrt not in nsrt_to_pos_lifted_atoms:
                nsrt_to_pos_lifted_atoms[nsrt] = []
            nsrt_to_pos_lifted_atoms[nsrt].append(pos_lifted_atoms)
            univ = {
                p
                for pred in self._predicates | self._failure_predicates
                for p in utils.get_all_ground_atoms_for_predicate(
                    pred, set(sub))
            }
            absent_atoms = univ - ground_atoms
            # Only consider negatives that were true at some point.
            absent_atoms &= all_seen_atoms
            neg_lifted_atoms = {
                a.lift(sub)
                for a in absent_atoms if all(o in sub for o in a.objects)
            }
            if nsrt not in nsrt_to_neg_lifted_atoms:
                nsrt_to_neg_lifted_atoms[nsrt] = []
            nsrt_to_neg_lifted_atoms[nsrt].append(neg_lifted_atoms)

        # Intersect to get positive state preconditions.
        # Also always include the preconditions of the NSRT itself.
        nsrt_to_pos_preconds = {
            nsrt: set.intersection(*atoms) | nsrt.preconditions
            for nsrt, atoms in nsrt_to_pos_lifted_atoms.items()
        }

        # Intersect to get negative state preconditions.
        nsrt_to_neg_preconds = {
            nsrt: set.intersection(*atoms)
            for nsrt, atoms in nsrt_to_neg_lifted_atoms.items()
        }

        # Create LDLRules and finish LiftedDecisionList.
        # The order is arbitrary.
        ldl_rules = []
        for nsrt in sorted(nsrt_to_pos_preconds):
            name = nsrt.name
            pos_preconds = nsrt_to_pos_preconds[nsrt]
            neg_preconds = nsrt_to_neg_preconds[nsrt]
            goal_preconds: Set[LiftedAtom] = set()  # not used
            rule = LDLRule(name, nsrt.parameters, pos_preconds, neg_preconds,
                           goal_preconds, nsrt)
            ldl_rules.append(rule)

        self._current_ldl = LiftedDecisionList(ldl_rules)
        logging.info("Learned bridge LDL:")
        logging.info(self._current_ldl)
