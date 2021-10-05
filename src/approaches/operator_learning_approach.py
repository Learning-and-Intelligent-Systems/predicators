"""A TAMP approach that learns operators and samplers.
"""

from collections import defaultdict
from typing import Any, Set, Tuple, List, Sequence
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, Operator, GroundAtom, _Option, \
    ParameterizedOption, LiftedAtom, Variable
from predicators.src import utils

Transition = Tuple[Set[GroundAtom], _Option, Set[GroundAtom], Set[GroundAtom]]


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
        print(f"\nLearning operators on {len(dataset)} trajectories...")

        # Set up data
        preds = self._initial_predicates
        transitions_by_option = defaultdict(list)
        for act_traj in dataset:
            states, options = utils.action_to_option_trajectory(act_traj)
            assert len(states) == len(options) + 1
            for i, option in enumerate(options):
                atoms = utils.abstract(states[i], preds)
                next_atoms = utils.abstract(states[i+1], preds)
                add_effects = next_atoms - atoms
                delete_effects = atoms - next_atoms
                transition = (atoms, option, add_effects, delete_effects)
                transitions_by_option[option.parent].append(transition)

        # Learn operators
        operators = []
        for param_option in transitions_by_option:
            option_transitions = transitions_by_option[param_option]
            option_ops = self._learn_operators_for_option(
                param_option, option_transitions)
            operators.extend(option_ops)

        print("Learned operators:")
        for operator in operators:
            print(operator)
        print()

        self._operators = set(operators)

    def _learn_operators_for_option(self, option: ParameterizedOption,
                                    transitions: List[Transition]
                                    ) -> List[Operator]:
        # Partition the data by lifted effects
        add_effects, delete_effects, partitioned_transitions = \
            self._partition_transitions_by_lifted_effects(transitions)

        # Learn preconditions
        operators = []
        for i, part_transitions in enumerate(partitioned_transitions):
            variables, preconditions = self._learn_preconditions(
                add_effects[i], delete_effects[i], part_transitions)
            # TODO: sampler learning
            sampler = lambda s, rng, objs: rng.uniform(size=(1,))
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
                if utils.unify(trans_add_effects, part_add_effects)[0] and \
                   utils.unify(trans_delete_effects, part_delete_effects)[0]:
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
            add_suc, add_sub = utils.unify(trans_add_effects, add_effects)
            assert add_suc  # else this transition won't be in this partition
            del_suc, del_sub = utils.unify(trans_delete_effects, delete_effects)
            assert del_suc  # else this transition won't be in this partition
            sub = add_sub | del_sub
            # Remove atoms from the state which contain objects not mentioned
            # in the effects. This cannot handle actions at a distance.
            objects = {o for atom in trans_add_effects |
                       trans_delete_effects for o in atom.objects}
            atoms = {atom for atom in atoms if
                     all(o in objects for o in atom.objects)}
            lifted_atoms = {atom.lift(sub) for atom in atoms}
            if i == 0:
                variables = sorted(sub.values())
            else:
                assert variables == sorted(sub.values())
            if i == 0:
                preconditions = lifted_atoms
            else:
                preconditions &= lifted_atoms

        return variables, preconditions
