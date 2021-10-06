"""A TAMP approach that learns operators and samplers.
"""

import functools
from collections import defaultdict
from typing import Any, Set, Tuple, List, Sequence, FrozenSet
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, Operator, GroundAtom, \
    ParameterizedOption, LiftedAtom, Variable, Predicate, ObjToVarSub, \
    Transition
from predicators.src import utils


class OperatorLearningApproach(TAMPApproach):
    """A TAMP approach that learns operators and samplers.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._operators: Set[Operator] = set()

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_operators(self) -> Set[Operator]:
        assert self._operators, "Operators not learned"
        return self._operators

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._operators = self.learn_operators_from_data(
            dataset, self._initial_predicates)

    @staticmethod
    def learn_operators_from_data(dataset: Dataset,
                                  predicates: Set[Predicate]
                                  ) -> Set[Operator]:
        """Learn operators from the given dataset of transitions.
        States are parsed using the given set of predicates.
        """
        print(f"\nLearning operators on {len(dataset)} trajectories...")

        # Set up data
        transitions_by_option = defaultdict(list)
        for act_traj in dataset:
            states, options = utils.action_to_option_trajectory(act_traj)
            assert len(states) == len(options) + 1
            for i, option in enumerate(options):
                atoms = utils.abstract(states[i], predicates)
                next_atoms = utils.abstract(states[i+1], predicates)
                add_effects = next_atoms - atoms
                delete_effects = atoms - next_atoms
                transition = (atoms, option, add_effects, delete_effects)
                transitions_by_option[option.parent].append(transition)

        # Learn operators
        operators = []
        for param_option in transitions_by_option:
            option_transitions = transitions_by_option[param_option]
            option_ops = OperatorLearningApproach._learn_operators_for_option(
                param_option, option_transitions)
            operators.extend(option_ops)

        print("Learned operators:")
        for operator in operators:
            print(operator)
        print()

        return set(operators)

    @staticmethod
    def _learn_operators_for_option(option: ParameterizedOption,
                                    transitions: List[Transition]
                                    ) -> List[Operator]:
        # Partition the data by lifted effects
        add_effects, delete_effects, partitioned_transitions = \
            OperatorLearningApproach._partition_transitions_by_lifted_effects(
                transitions)

        # Learn preconditions
        operators = []
        for i, part_transitions in enumerate(partitioned_transitions):
            variables, preconditions = \
                OperatorLearningApproach._learn_preconditions(
                    add_effects[i], delete_effects[i], part_transitions)
            sampler = lambda s, rng, objs: rng.uniform(
                size=option.params_space.shape)
            operators.append(Operator(
                f"{option.name}{i}", variables, preconditions,
                add_effects[i], delete_effects[i], option, sampler))

        return operators

    @staticmethod
    def _partition_transitions_by_lifted_effects(
            transitions: List[Transition]) -> Tuple[
                List[Set[LiftedAtom]], List[Set[LiftedAtom]],
                List[List[Transition]]]:
        add_effects: List[Set[LiftedAtom]] = []
        delete_effects: List[Set[LiftedAtom]] = []
        partitions: List[List[Transition]] = []
        num_partitions = 0
        for transition in transitions:
            _, _, trans_add_effects, trans_delete_effects = transition
            partition_index = None
            for i in range(num_partitions):
                # Try to unify this transition with existing effects
                # Note that both add and delete effects must unify
                part_add_effects = add_effects[i]
                part_delete_effects = delete_effects[i]
                if OperatorLearningApproach._unify(
                        frozenset(trans_add_effects),
                        frozenset(trans_delete_effects),
                        frozenset(part_add_effects),
                        frozenset(part_delete_effects))[0]:
                    # Add to this partition
                    partition_index = i
                    break
            # Otherwise, create a new group
            if partition_index is None:
                new_partition = [transition]
                partitions.append(new_partition)
                # Get new lifted effects
                objects = {o for atom in trans_add_effects |
                           trans_delete_effects for o in atom.objects}
                objects_lst = sorted(objects)
                variables = [Variable(f"?x{i}", o.type)
                             for i, o in enumerate(objects_lst)]
                sub = dict(zip(objects_lst, variables))
                add_effects.append({atom.lift(sub) for atom
                                    in trans_add_effects})
                delete_effects.append({atom.lift(sub) for atom
                                       in trans_delete_effects})
                num_partitions += 1
            # Add to existing group
            else:
                partitions[partition_index].append(transition)

        assert len(add_effects) == len(delete_effects) == len(partitions) == \
            num_partitions
        return add_effects, delete_effects, partitions

    @staticmethod
    def  _learn_preconditions(
            add_effects: Set[LiftedAtom], delete_effects: Set[LiftedAtom],
            transitions: List[Transition]) -> Tuple[
                Sequence[Variable], Set[LiftedAtom]]:
        for i, (atoms, _, trans_add_effects,
                trans_delete_effects) in enumerate(transitions):
            suc, sub = OperatorLearningApproach._unify(
                frozenset(trans_add_effects),
                frozenset(trans_delete_effects),
                frozenset(add_effects),
                frozenset(delete_effects))
            assert suc  # else this transition won't be in this partition
            # Remove atoms from the state which contain objects not mentioned
            # in the effects. This cannot handle actions at a distance.
            objects = {o for atom in trans_add_effects |
                       trans_delete_effects for o in atom.objects}
            atoms = {atom for atom in atoms if
                     all(o in objects for o in atom.objects)}
            lifted_atoms = {atom.lift(sub) for atom in atoms}
            if i == 0:
                variables = sorted(set(sub.values()))
            else:
                assert variables == sorted(set(sub.values()))
            if i == 0:
                preconditions = lifted_atoms
            else:
                preconditions &= lifted_atoms

        return variables, preconditions

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _unify(
            ground_add_effects: FrozenSet[GroundAtom],
            ground_delete_effects: FrozenSet[GroundAtom],
            lifted_add_effects: FrozenSet[LiftedAtom],
            lifted_delete_effects: FrozenSet[LiftedAtom]
    ) -> Tuple[bool, ObjToVarSub]:
        """Light wrapper around utils.unify() that handles split add and
        delete effects. Changes predicate names so that delete effects are
        treated differently than add effects by utils.unify().

        Note: We could only change either add or delete predicate names,
        but to avoid potential bugs we'll just change both.
        """
        new_ground_add_effects = set()
        for ground_atom in ground_add_effects:
            new_predicate = Predicate("ADD-"+ground_atom.predicate.name,
                                      ground_atom.predicate.types,
                                      _classifier=lambda s, o: False)  # dummy
            new_ground_add_effects.add(GroundAtom(
                new_predicate, ground_atom.objects))
        f_new_ground_add_effects = frozenset(new_ground_add_effects)
        new_ground_delete_effects = set()
        for ground_atom in ground_delete_effects:
            new_predicate = Predicate("DEL-"+ground_atom.predicate.name,
                                      ground_atom.predicate.types,
                                      _classifier=lambda s, o: False)  # dummy
            new_ground_delete_effects.add(GroundAtom(
                new_predicate, ground_atom.objects))
        f_new_ground_delete_effects = frozenset(new_ground_delete_effects)
        new_lifted_add_effects = set()
        for lifted_atom in lifted_add_effects:
            new_predicate = Predicate("ADD-"+lifted_atom.predicate.name,
                                      lifted_atom.predicate.types,
                                      _classifier=lambda s, o: False)  # dummy
            new_lifted_add_effects.add(LiftedAtom(
                new_predicate, lifted_atom.variables))
        f_new_lifted_add_effects = frozenset(new_lifted_add_effects)
        new_lifted_delete_effects = set()
        for lifted_atom in lifted_delete_effects:
            new_predicate = Predicate("DEL-"+lifted_atom.predicate.name,
                                      lifted_atom.predicate.types,
                                      _classifier=lambda s, o: False)  # dummy
            new_lifted_delete_effects.add(LiftedAtom(
                new_predicate, lifted_atom.variables))
        f_new_lifted_delete_effects = frozenset(new_lifted_delete_effects)
        return utils.unify(
            f_new_ground_add_effects | f_new_ground_delete_effects,
            f_new_lifted_add_effects | f_new_lifted_delete_effects)
