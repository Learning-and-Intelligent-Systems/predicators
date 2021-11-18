"""Algorithms for learning the various components of NSRT objects.
"""

import functools
from collections import defaultdict
from typing import Set, Tuple, List, Sequence, FrozenSet, DefaultDict
from predicators.src.structs import Dataset, STRIPSOperator, NSRT, \
    GroundAtom, ParameterizedOption, LiftedAtom, Variable, Predicate, \
    ObjToVarSub, Transition, Object
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.sampler_learning import learn_sampler


def learn_nsrts_from_data(dataset: Dataset, predicates: Set[Predicate],
                          do_sampler_learning: bool) -> Set[NSRT]:
    """Learn NSRTs from the given dataset of transitions.
    States are abstracted using the given set of predicates.
    """
    print(f"\nLearning NSRTs on {len(dataset)} trajectories...")

    transitions_by_option = generate_transitions(dataset, predicates)

    nsrts = []
    for param_option in transitions_by_option:
        option_transitions = transitions_by_option[param_option]
        option_nsrts = learn_nsrts_for_option(
            param_option, option_transitions, do_sampler_learning)
        nsrts.extend(option_nsrts)

    print("\nLearned NSRTs:")
    for nsrt in nsrts:
        print(nsrt)
    print()

    return set(nsrts)


def generate_transitions(dataset: Dataset, predicates: Set[Predicate]
                         ) -> DefaultDict[
                             ParameterizedOption, List[Transition]]:
    """Given a dataset and predicates, go through the dataset and compute
    the abstract states. Returns a dict mapping ParameterizedOptions to a
    list of transitions.
    """
    transitions_by_option = defaultdict(list)
    for act_traj in dataset:
        states, options = utils.action_to_option_trajectory(act_traj)
        assert len(states) == len(options) + 1
        for i, option in enumerate(options):
            atoms = utils.abstract(states[i], predicates)
            next_atoms = utils.abstract(states[i+1], predicates)
            add_effects = next_atoms - atoms
            delete_effects = atoms - next_atoms
            transition = (states[i], states[i+1], atoms, option, next_atoms,
                          add_effects, delete_effects)
            transitions_by_option[option.parent].append(transition)
    return transitions_by_option


def learn_nsrts_for_option(option: ParameterizedOption,
                               transitions: List[Transition],
                               do_sampler_learning: bool,
                               ) -> List[NSRT]:
    """Given an option and data for it, learn NSRTs.
    """
    # Partition the data by lifted effects
    option_vars, add_effects, delete_effects, \
        partitioned_transitions = _partition_transitions(transitions)

    nsrts = []
    for i, part_transitions in enumerate(partitioned_transitions):
        if len(part_transitions) < CFG.min_data_for_nsrt:
            continue
        if not add_effects[i] and not delete_effects[i]:
            # Don't learn any NSRTs for empty effects, since they're
            # not useful for planning or predicate invention.
            continue
        # Learn preconditions
        variables, preconditions = \
            _learn_preconditions(option_vars[i], add_effects[i],
                                 delete_effects[i], part_transitions)
        name = f"{option.name}{i}"
        strips_operator = STRIPSOperator(
            name, variables, preconditions,
            add_effects[i], delete_effects[i])
        # Learn sampler
        sampler = learn_sampler(
            partitioned_transitions, name, variables, preconditions,
            add_effects[i], delete_effects[i], option, i, do_sampler_learning)
        # Construct NSRT object
        nsrts.append(strips_operator.make_operator(
            option, option_vars[i], sampler))

    return nsrts


def _partition_transitions(
        transitions: List[Transition]) -> Tuple[
            List[List[Variable]],
            List[Set[LiftedAtom]],
            List[Set[LiftedAtom]],
            List[List[Tuple[Transition, ObjToVarSub]]]]:
    option_args: List[List[Variable]] = []
    add_effects: List[Set[LiftedAtom]] = []
    delete_effects: List[Set[LiftedAtom]] = []
    partitions: List[List[Tuple[Transition, ObjToVarSub]]] = []
    for transition in transitions:
        _, _, _, option, _, trans_add_effects, trans_delete_effects = transition
        trans_option_args = option.objects
        for i in range(len(partitions)):
            # Try to unify this transition with existing effects
            # Note that both add and delete effects must unify
            part_option_args = option_args[i]
            part_add_effects = add_effects[i]
            part_delete_effects = delete_effects[i]
            suc, sub = _unify(frozenset(trans_add_effects),
                              frozenset(trans_delete_effects),
                              tuple(trans_option_args),
                              frozenset(part_add_effects),
                              frozenset(part_delete_effects),
                              tuple(part_option_args))
            if suc:
                # Add to this partition
                partitions[i].append((transition, sub))
                break
        # Otherwise, create a new group
        else:
            # Get new lifted effects
            objects = {o for atom in trans_add_effects |
                       trans_delete_effects for o in atom.objects}
            objects.update(option.objects)
            objects_lst = sorted(objects)
            variables = [Variable(f"?x{i}", o.type)
                         for i, o in enumerate(objects_lst)]
            sub = dict(zip(objects_lst, variables))
            option_args.append([sub[v] for v in trans_option_args])
            add_effects.append({atom.lift(sub) for atom
                                in trans_add_effects})
            delete_effects.append({atom.lift(sub) for atom
                                   in trans_delete_effects})
            new_partition = [(transition, sub)]
            partitions.append(new_partition)

    assert len(option_args) == len(add_effects) == \
           len(delete_effects) == len(partitions)
    return option_args, add_effects, delete_effects, partitions


def  _learn_preconditions(option_vars: List[Variable],
        add_effects: Set[LiftedAtom], delete_effects: Set[LiftedAtom],
        transitions: List[Tuple[Transition, ObjToVarSub]]) -> Tuple[
            Sequence[Variable], Set[LiftedAtom]]:
    for i, ((_, _, atoms, option, _, trans_add_effects,
             trans_delete_effects), _) in enumerate(transitions):
        suc, sub = _unify(
            frozenset(trans_add_effects),
            frozenset(trans_delete_effects),
            tuple(option.objects),
            frozenset(add_effects),
            frozenset(delete_effects),
            tuple(option_vars))
        assert suc  # else this transition won't be in this partition
        # Remove atoms from the state which contain objects not mentioned
        # in the effects or option. This cannot handle actions at a distance.
        objects = {o for atom in trans_add_effects |
                   trans_delete_effects for o in atom.objects}
        objects.update(option.objects)
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


@functools.lru_cache(maxsize=None)
def _unify(
        ground_add_effects: FrozenSet[GroundAtom],
        ground_delete_effects: FrozenSet[GroundAtom],
        ground_option_args: Tuple[Object, ...],
        lifted_add_effects: FrozenSet[LiftedAtom],
        lifted_delete_effects: FrozenSet[LiftedAtom],
        lifted_option_args: Tuple[Variable, ...]
) -> Tuple[bool, ObjToVarSub]:
    """Wrapper around utils.unify() that handles option arguments, add effects,
    and delete effects. Changes predicate names so that all are treated
    differently by utils.unify().
    """
    opt_arg_pred = Predicate("OPT-ARGS",
                             [a.type for a in ground_option_args],
                             _classifier=lambda s, o: False)  # dummy
    f_ground_option_args = frozenset({GroundAtom(opt_arg_pred,
                                                 ground_option_args)})
    new_ground_add_effects = utils.wrap_atom_predicates_ground(
        ground_add_effects, "ADD-")
    f_new_ground_add_effects = frozenset(new_ground_add_effects)
    new_ground_delete_effects = utils.wrap_atom_predicates_ground(
        ground_delete_effects, "DEL-")
    f_new_ground_delete_effects = frozenset(new_ground_delete_effects)

    f_lifted_option_args = frozenset({LiftedAtom(opt_arg_pred,
                                                 lifted_option_args)})
    new_lifted_add_effects = utils.wrap_atom_predicates_lifted(
        lifted_add_effects, "ADD-")
    f_new_lifted_add_effects = frozenset(new_lifted_add_effects)
    new_lifted_delete_effects = utils.wrap_atom_predicates_lifted(
        lifted_delete_effects, "DEL-")
    f_new_lifted_delete_effects = frozenset(new_lifted_delete_effects)
    return utils.unify(
        f_ground_option_args | f_new_ground_add_effects | \
            f_new_ground_delete_effects,
        f_lifted_option_args | f_new_lifted_add_effects | \
            f_new_lifted_delete_effects)
