"""Code for learning the STRIPS operators within NSRTs."""

import logging
from typing import List, Sequence, Set, cast

from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.structs import DummyOption, LiftedAtom, \
    PartialNSRTAndDatastore, Predicate, Segment, STRIPSOperator, Variable, \
    VarToObjSub


def learn_strips_operators(
    segments: Sequence[Segment],
    verbose: bool = True,
) -> List[PartialNSRTAndDatastore]:
    """Learn strips operators on the given data segments.

    Return a list of PNADs with op (STRIPSOperator), datastore, and
    option_spec fields filled in.
    """
    # Cluster the segments according to common effects.
    pnads: List[PartialNSRTAndDatastore] = []
    for segment in segments:
        if segment.has_option():
            segment_option = segment.get_option()
            segment_param_option = segment_option.parent
            segment_option_objs = tuple(segment_option.objects)
        else:
            segment_param_option = DummyOption.parent
            segment_option_objs = tuple()
        for pnad in pnads:
            # Try to unify this transition with existing effects.
            # Note that both add and delete effects must unify,
            # and also the objects that are arguments to the options.
            (pnad_param_option, pnad_option_vars) = pnad.option_spec
            suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                frozenset(),  # no preconditions
                frozenset(),  # no preconditions
                frozenset(segment.add_effects),
                frozenset(pnad.op.add_effects),
                frozenset(segment.delete_effects),
                frozenset(pnad.op.delete_effects),
                segment_param_option,
                pnad_param_option,
                segment_option_objs,
                tuple(pnad_option_vars))
            sub = cast(VarToObjSub, {v: o for o, v in ent_to_ent_sub.items()})
            if suc:
                # Add to this PNAD.
                assert set(sub.keys()) == set(pnad.op.parameters)
                pnad.add_to_datastore((segment, sub))
                break
        else:
            # Otherwise, create a new PNAD.
            objects = {o for atom in segment.add_effects |
                       segment.delete_effects for o in atom.objects} | \
                      set(segment_option_objs)
            objects_lst = sorted(objects)
            params = [
                Variable(f"?x{i}", o.type) for i, o in enumerate(objects_lst)
            ]
            preconds: Set[LiftedAtom] = set()  # will be learned later
            obj_to_var = dict(zip(objects_lst, params))
            var_to_obj = dict(zip(params, objects_lst))
            add_effects = {
                atom.lift(obj_to_var)
                for atom in segment.add_effects
            }
            delete_effects = {
                atom.lift(obj_to_var)
                for atom in segment.delete_effects
            }
            side_predicates: Set[Predicate] = set()  # will be learned later
            op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                add_effects, delete_effects, side_predicates)
            datastore = [(segment, var_to_obj)]
            option_vars = [obj_to_var[o] for o in segment_option_objs]
            option_spec = (segment_param_option, option_vars)
            pnads.append(PartialNSRTAndDatastore(op, datastore, option_spec))

    # Prune PNADs with not enough data.
    pnads = [
        pnad for pnad in pnads if len(pnad.datastore) >= CFG.min_data_for_nsrt
    ]

    # Learn the preconditions of the operators in the PNADs via intersection.
    for pnad in pnads:
        preconditions = induce_pnad_preconditions(pnad)
        # Replace the operator with one that contains the newly learned
        # preconditions. We do this because STRIPSOperator objects are
        # frozen, so their fields cannot be modified.
        pnad.op = STRIPSOperator(pnad.op.name, pnad.op.parameters,
                                 preconditions, pnad.op.add_effects,
                                 pnad.op.delete_effects,
                                 pnad.op.side_predicates)

    # Print and return the PNADs.
    if verbose:
        logging.info("Learned operators (before side predicate & option "
                     "learning):")
        for pnad in pnads:
            logging.info(pnad)
    return pnads


def induce_pnad_preconditions(
        pnad: PartialNSRTAndDatastore) -> Set[LiftedAtom]:
    """Given a PNAD with a nonempty datastore, compute the preconditions for
    the PNAD operator by intersecting all lifted preimages in the datastore."""
    assert len(pnad.datastore) > 0
    for i, (segment, var_to_obj) in enumerate(pnad.datastore):
        objects = set(var_to_obj.values())
        obj_to_var = {o: v for v, o in var_to_obj.items()}
        atoms = {
            atom
            for atom in segment.init_atoms
            if all(o in objects for o in atom.objects)
        }
        lifted_atoms = {atom.lift(obj_to_var) for atom in atoms}
        if i == 0:
            preconditions = lifted_atoms
        else:
            preconditions &= lifted_atoms
    return preconditions
