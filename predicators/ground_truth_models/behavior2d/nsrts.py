"""Ground-truth NSRTs for the Behavior2D environment."""

import itertools
from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler

"""
Run this piece of code to get all relevant NextTo pairs so that we don't need to enumerate them all

python get_nextto_type_pairs.py
"""

# NEXTTO_TYPES = {'carrot', 'gingerbread', 'sandal', 'dustpan', 'bath_towel', 'cupcake', 'tray', 
#     'hairbrush', 'fish', 'container_date', 'coffee_table', 'spoon', 'salad', 'soap', 'muffin', 
#     'hand_towel', 'countertop', 'toothbrush', 'sink', 'rag', 'shelf', 'baguette', 'cookie', 'paper_towel', 
#     'paintbrush', 'dishtowel', 'bucket', 'bagel', 'vacuum', 'bed', 'fridge', 'gym_shoe', 'soup', 'plate',
#     'olive', 'pretzel', 'orange', 'apple', 'bowl', 'scrub_brush', 'breakfast_table'}
NEXTTO_TYPE_PAIRS = {('cupcake', 'plate'), ('paper_towel', 'sink'), ('wrapped_gift', 'christmas_tree'), 
                     ('scrub_brush', 'paper_towel'), ('bow', 'bottom_cabinet'), ('sofa', 'window'), 
                     ('scrub_brush', 'dishtowel'), ('hardback', 'basket'), ('rag', 'sink'), 
                     ('highchair', 'window'), ('sheet', 'coffee_table'), ('gym_shoe', 'breakfast_table'), 
                     ('dishtowel', 'sink'), ('toothbrush', 'hand_towel'), ('cookie', 'plate'), 
                     ('notebook', 'backpack'), ('spoon', 'soup'), ('armchair', 'window'), ('pork', 'pork'), 
                     ('floor_lamp', 'window'), ('watermelon', 'carton'), ('floor_lamp', 'door'), 
                     ('salad', 'plate'), ('folder', 'hardback'), ('pretzel', 'plate'), 
                     ('hairbrush', 'hand_towel'), ('baguette', 'plate'), ('scrub_brush', 'bath_towel'), 
                     ('folder', 'notebook'), ('gingerbread', 'plate'), ('spoon', 'sink'), 
                     ('cauldron', 'coffee_table'), ('chaise_longue', 'window'), ('folder', 'backpack'), 
                     ('soap', 'sink'), ('paintbrush', 'hand_towel'), ('cantaloup', 'carton'), 
                     ('tray', 'fridge'), ('toothbrush', 'paper_towel'), ('wreath', 'bottom_cabinet'), 
                     ('salad', 'pretzel'), ('table_lamp', 'window'), ('bucket', 'countertop'), 
                     ('table_lamp', 'door'), ('bowl', 'sink'), ('fish', 'sink'), ('gym_shoe', 'coffee_table'), 
                     ('stool', 'window'), ('wrapped_gift', 'christmas_tree_decorated'), 
                     ('swivel_chair', 'window'), ('hairbrush', 'paper_towel'), ('paintbrush', 'bath_towel'), 
                     ('hairbrush', 'dishtowel'), ('paintbrush', 'paper_towel'), ('bagel', 'plate'), 
                     ('orange', 'orange'), ('bucket', 'sink'), ('bench', 'window'), 
                     ('toothbrush', 'bath_towel'), ('carrot', 'carrot'), ('vacuum', 'bed'), 
                     ('paintbrush', 'dishtowel'), ('broccoli', 'broccoli'), ('sandal', 'shelf'), 
                     ('hardback', 'backpack'), ('toothbrush', 'dishtowel'), ('sheet', 'breakfast_table'), 
                     ('rocking_chair', 'window'), ('apple', 'apple'), ('broom', 'sink'), ('cereal', 'cereal'), 
                     ('scrub_brush', 'hand_towel'), ('hairbrush', 'bath_towel'), ('notebook', 'basket'), 
                     ('raspberry', 'raspberry'), ('muffin', 'plate'), ('olive', 'sink'), ('hand_towel', 'sink'), 
                     ('dustpan', 'fridge'), ('bath_towel', 'sink'), ('folding_chair', 'window'), 
                     ('cauldron', 'breakfast_table'), ('mousetrap', 'toilet'), ('lettuce', 'lettuce'), 
                     ('container_date', 'fish'), ('backpack', 'bed'), ('straight_chair', 'window')}


TOUCHING_TYPE_PAIRS = {('stool', 'bed'), ('straight_chair', 'bed'), ('folding_chair', 'bed'), 
                       ('swivel_chair', 'bed'), ('rocking_chair', 'bed'), ('highchair', 'bed'), ('bench', 'bed'), 
                       ('armchair', 'bed'), ('sofa', 'bed'), ('envelope', 'envelope'), ('newspaper', 'newspaper'), 
                       ('chaise_longue', 'bed')}


class Behavior2DGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Behavior2D environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"behavior2d"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        robot_type = types["robot"]
        robot_obj = Variable("?robby", robot_type)
        other_types = list(set(named_type.parent for name, named_type in types.items() if name != "robot" and name != "object"))
        # other_types = [named_type for name, named_type in types.items() if name != "robot"]
        assert len(other_types) == 1
        object_type = other_types[0]

        handempty = LiftedAtom(predicates["handempty"], [])


        nsrts = set()
        
        # NavigateTo
        def navigate_to_param_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objects: Sequence[Object]) -> Array:
            del goal
            _, obj = objects
            closeness_limit = 2.00
            nearness_limit = 0.15
            distance = nearness_limit + (closeness_limit - nearness_limit) * rng.random()
            yaw = rng.random() * (2 * np.pi) - np.pi
            x = distance * np.cos(yaw)
            y = distance * np.sin(yaw)

            obj_w = state.get(obj, "width")
            obj_h = state.get(obj, "height")

            x = x + obj_w / 2   # logic for BEHAVIOR sampling assumes center of object, but Behavior2D uses bottom left
            y = y + obj_h / 2   # logic for BEHAVIOR sampling assumes center of object, but Behavior2D uses bottom left

            return np.array([x, y])
        target_obj = Variable("?targ", object_type)

        parameters = [robot_obj, target_obj]
        option_vars = [robot_obj, target_obj]
        preconditions = {LiftedAtom(predicates["not-holding"], [target_obj])}
        add_effects = {LiftedAtom(predicates["reachable"], [target_obj])}
        delete_effects = {LiftedAtom(predicates["reachable-nothing"], [])}
        nsrt = NSRT(
            "NavigateTo", parameters,
            preconditions, add_effects, delete_effects,
            {predicates["reachable"]}, options["NavigateTo"], option_vars,
            navigate_to_param_sampler
        )
        nsrts.add(nsrt)

        # Grasp
        def grasp_obj_param_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objects: Sequence[Object]) -> Array:
            del state, goal, objects
            offset = rng.random()
            yaw = rng.random() * (2 * np.pi) - np.pi

            return np.array([offset, yaw])
        target_obj = Variable("?targ", object_type)
        surf_obj = Variable("?surf", object_type)
        parameters = [robot_obj, target_obj]
        option_vars = [robot_obj, target_obj]
        targ_reachable = LiftedAtom(predicates["reachable"], [target_obj])
        targ_holding = LiftedAtom(predicates["holding"], [target_obj])
        target_not_holding = LiftedAtom(predicates["not-holding"], [target_obj])
        
        inside = (predicates["inside"], {0: target_obj})
        nextto_0 = (predicates["nextto"], {0: target_obj})
        nextto_1 = (predicates["nextto"], {1: target_obj})

        preconditions = {handempty, targ_reachable}
        add_effects = {targ_holding}
        delete_effects = {handempty, targ_reachable, target_not_holding}
        nsrt = NSRT(
            "Grasp", parameters,
            preconditions, add_effects, delete_effects,
            set(), options["Grasp"], option_vars, grasp_obj_param_sampler
        )
        nsrt.add_fancy_ignore_effect(*inside)
        nsrt.add_fancy_ignore_effect(*nextto_0)
        nsrt.add_fancy_ignore_effect(*nextto_1)
        nsrts.add(nsrt)

        # PlaceInside
        def place_inside_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objects: Sequence[Object]) -> Array:
            del state, goal, objects
            offset = rng.random()

            return np.array([offset])
        surf_obj = Variable("?surf", object_type)

        held_obj = Variable("?held", object_type)
        parameters = [robot_obj, held_obj, surf_obj]
        option_vars = [robot_obj, surf_obj]
        held_holding = LiftedAtom(predicates["holding"], [held_obj])
        held_nothodling = LiftedAtom(predicates["not-holding"], [held_obj])
        surf_reachable = LiftedAtom(predicates["reachable"], [surf_obj])
        held_reachable = LiftedAtom(predicates["reachable"], [held_obj])
        insideable = LiftedAtom(predicates["insideable"], [surf_obj])
        inside = LiftedAtom(predicates["inside"], [held_obj, surf_obj])
        inside_nothing = LiftedAtom(predicates["inside-nothing"], [held_obj])
        not_openable_surf = LiftedAtom(predicates["not-openable"], [surf_obj])
        open_surf = LiftedAtom(predicates["open"], [surf_obj])
        preconditions_openable = {insideable, held_holding, surf_reachable, open_surf}
        preconditions_not_openable = {insideable, held_holding, surf_reachable, not_openable_surf}
        add_effects = {inside, handempty, held_reachable, held_nothodling}
        delete_effects = {held_holding, inside_nothing}
        nsrt = NSRT(
            "PlaceInside-openable", parameters,
            preconditions_openable, add_effects, delete_effects,
            set(), options["PlaceInside"], option_vars, place_inside_sampler
        )
        nsrts.add(nsrt)
        nsrt = NSRT(
            "PlaceInside-notOpenable", parameters,
            preconditions_not_openable, add_effects, delete_effects,
            set(), options["PlaceInside"], option_vars, place_inside_sampler
        )
        nsrts.add(nsrt)

        # Note: technically, we could skip this loop and just create a generic
        # PlaceNextTo operator for the parent object_type. However, that leads
        # FastDownward to create one grounding for each triple of object types,
        # which can result in several million operators.
        for held_obj_type_name, nextto_type_name in NEXTTO_TYPE_PAIRS:
            if held_obj_type_name not in types or nextto_type_name not in types:
                continue
            held_obj_type = types[held_obj_type_name]
            nextto_type = types[nextto_type_name]
            held_obj = Variable("?held", held_obj_type)
            nextto_obj = Variable("?nextto", nextto_type)
            parameters = [robot_obj, held_obj, surf_obj, nextto_obj]
            option_vars = [robot_obj, surf_obj]
            nextto = LiftedAtom(predicates["nextto"], [held_obj, nextto_obj])
            held_holding = LiftedAtom(predicates["holding"], [held_obj])
            held_nothodling = LiftedAtom(predicates["not-holding"], [held_obj])
            held_reachable = LiftedAtom(predicates["reachable"], [held_obj])
            inside = LiftedAtom(predicates["inside"], [held_obj, surf_obj])
            inside_nothing = LiftedAtom(predicates["inside-nothing"], [held_obj])
            nextto_inside = LiftedAtom(predicates["inside"], [nextto_obj, surf_obj])
            preconditions_openable = {held_holding, surf_reachable, open_surf, nextto_inside}
            preconditions_not_openable = {held_holding, surf_reachable, not_openable_surf, nextto_inside}
            add_effects = {inside, handempty, held_reachable, nextto, held_nothodling}
            delete_effects = {held_holding, inside_nothing}
            nsrt = NSRT(
                f"PlaceInsideNextTo-{held_obj_type_name}-{nextto_type_name}-openable", parameters,
                preconditions_openable, add_effects, delete_effects,
                set(), options["PlaceInside"], option_vars, place_inside_sampler
            )
            nsrts.add(nsrt)
            nsrt = NSRT(
                f"PlaceInsideNextTo-{held_obj_type_name}-{nextto_type_name}-notOpenable", parameters,
                preconditions_not_openable, add_effects, delete_effects,
                set(), options["PlaceInside"], option_vars, place_inside_sampler
            )
            nsrts.add(nsrt) 

        # PlaceNextTo (not Inside)
        for held_obj_type_name, nextto_type_name in NEXTTO_TYPE_PAIRS:
            if held_obj_type_name not in types or nextto_type_name not in types:
                continue
            held_obj_type = types[held_obj_type_name]
            nextto_type = types[nextto_type_name]
            held_obj = Variable("?held", held_obj_type)
            nextto_obj = Variable("?nextto", nextto_type)
            parameters = [robot_obj, held_obj, nextto_obj]
            option_vars = [robot_obj, nextto_obj]
            nextto = LiftedAtom(predicates["nextto"], [held_obj, nextto_obj])
            held_holding = LiftedAtom(predicates["holding"], [held_obj])
            held_nothodling = LiftedAtom(predicates["not-holding"], [held_obj])
            held_reachable = LiftedAtom(predicates["reachable"], [held_obj])
            nextto_reachable = LiftedAtom(predicates["reachable"], [nextto_obj])
            inside_nothing = LiftedAtom(predicates["inside-nothing"], [nextto_obj])
            preconditions = {held_holding, nextto_reachable, inside_nothing}
            add_effects = {handempty, held_reachable, nextto, held_nothodling}
            delete_effects = {held_holding}
            nsrt = NSRT(
                f"PlaceNextTo-{held_obj_type_name}-{nextto_type_name}-notInside", parameters,
                preconditions, add_effects, delete_effects, 
                set(), options["PlaceNextTo"], option_vars, place_inside_sampler
            )
            nsrts.add(nsrt)

        # Note: technically, we could skip this loop and just create a generic
        # PlaceTouching operator for the parent object_type. However, that leads
        # FastDownward to create one grounding for each triple of object types,
        # which can result in several million operators.
        for held_obj_type_name, touching_type_name in TOUCHING_TYPE_PAIRS:
            if held_obj_type_name not in types or touching_type_name not in types:
                continue
            held_obj_type = types[held_obj_type_name]
            touching_type = types[touching_type_name]
            held_obj = Variable("?held", held_obj_type)
            touching_obj = Variable("?touching", touching_type)
            parameters = [robot_obj, held_obj, surf_obj, touching_obj]
            option_vars = [robot_obj, surf_obj]
            touching = LiftedAtom(predicates["touching"], [held_obj, touching_obj])
            held_holding = LiftedAtom(predicates["holding"], [held_obj])
            held_nothodling = LiftedAtom(predicates["not-holding"], [held_obj])
            held_reachable = LiftedAtom(predicates["reachable"], [held_obj])
            inside = LiftedAtom(predicates["inside"], [held_obj, surf_obj])
            inside_nothing = LiftedAtom(predicates["inside-nothing"], [held_obj])
            touching_inside = LiftedAtom(predicates["inside"], [touching_obj, surf_obj])
            preconditions_openable = {held_holding, surf_reachable, open_surf, touching_inside}
            preconditions_not_openable = {held_holding, surf_reachable, not_openable_surf, touching_inside}
            add_effects = {inside, handempty, held_reachable, touching, held_nothodling}
            delete_effects = {held_holding, inside_nothing}
            nsrt = NSRT(
                f"PlaceInsideTouching-{held_obj_type_name}-{touching_type_name}-openable", parameters,
                preconditions_openable, add_effects, delete_effects,
                set(), options["PlaceInside"], option_vars, place_inside_sampler
            )
            nsrts.add(nsrt)
            nsrt = NSRT(
                f"PlaceInsideTouching-{held_obj_type_name}-{touching_type_name}-notOpenable", parameters,
                preconditions_not_openable, add_effects, delete_effects,
                set(), options["PlaceInside"], option_vars, place_inside_sampler
            )
            nsrts.add(nsrt) 

        # PlaceTouching (not Inside)
        for held_obj_type_name, touching_type_name in TOUCHING_TYPE_PAIRS:
            if held_obj_type_name not in types or touching_type_name not in types:
                continue
            held_obj_type = types[held_obj_type_name]
            touching_type = types[touching_type_name]
            held_obj = Variable("?held", held_obj_type)
            touching_obj = Variable("?touching", touching_type)
            parameters = [robot_obj, held_obj, touching_obj]
            option_vars = [robot_obj, touching_obj]
            touching = LiftedAtom(predicates["touching"], [held_obj, touching_obj])
            held_holding = LiftedAtom(predicates["holding"], [held_obj])
            held_nothodling = LiftedAtom(predicates["not-holding"], [held_obj])
            held_reachable = LiftedAtom(predicates["reachable"], [held_obj])
            touching_reachable = LiftedAtom(predicates["reachable"], [touching_obj])
            inside_nothing = LiftedAtom(predicates["inside-nothing"], [touching_obj])
            preconditions = {held_holding, touching_reachable, inside_nothing}
            add_effects = {handempty, held_reachable, touching, held_nothodling}
            delete_effects = {held_holding}
            nsrt = NSRT(
                f"PlaceTouching-{held_obj_type_name}-{touching_type_name}-notInside", parameters,
                preconditions, add_effects, delete_effects, 
                set(), options["PlaceTouching"], option_vars, place_inside_sampler
            )
            nsrts.add(nsrt)

        # Open
        open_obj = Variable("?obj", object_type)
        parameters = [robot_obj, open_obj]
        option_vars = [robot_obj, open_obj]
        openable = LiftedAtom(predicates["openable"], [open_obj])
        closed = LiftedAtom(predicates["closed"], [open_obj])
        reachable = LiftedAtom(predicates["reachable"], [open_obj])
        preconditions = {reachable, closed, openable, handempty}
        add_effects = {LiftedAtom(predicates["open"], [open_obj])}
        delete_effects = {closed}
        nsrt = NSRT(
            f"Open", parameters,
            preconditions, add_effects, delete_effects, set(),
            options["Open"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # Close
        close_obj = Variable("?obj", object_type)
        parameters = [robot_obj, close_obj]
        option_vars = [robot_obj, close_obj]
        openable = LiftedAtom(predicates["openable"], [close_obj])
        open_ = LiftedAtom(predicates["open"], [close_obj])
        reachable = LiftedAtom(predicates["reachable"], [close_obj])
        preconditions = {reachable, open_, openable, handempty}
        add_effects = {LiftedAtom(predicates["closed"], [close_obj])}
        delete_effects = {open_}
        nsrt = NSRT(
            f"Close", parameters,
            preconditions, add_effects, delete_effects, set(),
            options["Close"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # ToggleOn
        toggle_obj = Variable("?obj", object_type)
        parameters = [robot_obj, toggle_obj]
        option_vars = [robot_obj, toggle_obj]
        toggleable = LiftedAtom(predicates["toggleable"], [toggle_obj])
        toggledoff = LiftedAtom(predicates["toggled_off"], [toggle_obj])
        reachable = LiftedAtom(predicates["reachable"], [toggle_obj])
        preconditions = {reachable, toggledoff, toggleable, handempty}
        add_effects = {LiftedAtom(predicates["toggled_on"], [toggle_obj])}
        delete_effects = {toggledoff}
        nsrt = NSRT(
            f"ToggleOn", parameters,
            preconditions, add_effects, delete_effects, set(),
            options["ToggleOn"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # Cook
        cook_obj = Variable("?obj", object_type)
        cooker_obj = Variable("?cooker", object_type)
        parameters = [robot_obj, cooker_obj, cook_obj]
        option_vars = [robot_obj, cooker_obj]
        reachable = LiftedAtom(predicates["reachable"], [cooker_obj])
        holding_held = LiftedAtom(predicates["holding"], [cook_obj])
        cookable = LiftedAtom(predicates["cookable"], [cook_obj])
        cooker = LiftedAtom(predicates["cooker"], [cooker_obj])
        not_toggleable = LiftedAtom(predicates["not-toggleable"], [cooker_obj])
        toggled = LiftedAtom(predicates["toggled_on"], [cooker_obj])
        preconditions_toggleable = {reachable, holding_held, cookable, cooker, toggled}
        preconditions_not_toggleable = {reachable, holding_held, cookable, cooker, not_toggleable}
        add_effects = {LiftedAtom(predicates["cooked"], [cook_obj])}
        delete_effects = set()
        nsrt = NSRT(
            "Cook-toggleable", parameters,
            preconditions_toggleable, add_effects, delete_effects, set(),
            options["Cook"], option_vars, null_sampler
        )
        nsrts.add(nsrt)
        nsrt = NSRT(
            "Cook-notToggleable", parameters,
            preconditions_not_toggleable, add_effects, delete_effects, set(),
            options["Cook"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # Freeze
        freeze_obj = Variable("?obj", object_type)
        freezer_obj = Variable("?freezer", object_type)
        parameters = [robot_obj, freezer_obj, freeze_obj]
        option_vars = [robot_obj, freezer_obj]
        reachable = LiftedAtom(predicates["reachable"], [freezer_obj])
        holding_held = LiftedAtom(predicates["holding"], [freeze_obj])
        freezable = LiftedAtom(predicates["freezable"], [freeze_obj])
        freezer = LiftedAtom(predicates["freezer"], [freezer_obj])
        not_toggleable = LiftedAtom(predicates["not-toggleable"], [freezer_obj])
        toggled = LiftedAtom(predicates["toggled_on"], [freezer_obj])
        preconditions_toggleable = {reachable, holding_held, freezable, freezer, toggled}
        preconditions_not_toggleable = {reachable, holding_held, freezable, freezer, not_toggleable}
        add_effects = {LiftedAtom(predicates["frozen"], [freeze_obj])}
        delete_effects = set()
        nsrt = NSRT(
            "Freeze-toggleable", parameters,
            preconditions_toggleable, add_effects, delete_effects, set(),
            options["Freeze"], option_vars, null_sampler
        )
        nsrts.add(nsrt)
        nsrt = NSRT(
            "Freeze-notToggleable", parameters,
            preconditions_not_toggleable, add_effects, delete_effects, set(),
            options["Freeze"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # Soak
        soak_obj = Variable("?obj", object_type)
        soaker_obj = Variable("?soaker", object_type)
        parameters = [robot_obj, soaker_obj, soak_obj]
        option_vars = [robot_obj, soaker_obj]
        reachable = LiftedAtom(predicates["reachable"], [soaker_obj])
        holding_held = LiftedAtom(predicates["holding"], [soak_obj])
        soakable = LiftedAtom(predicates["soakable"], [soak_obj])
        soaker = LiftedAtom(predicates["soaker"], [soaker_obj])
        not_toggleable = LiftedAtom(predicates["not-toggleable"], [soaker_obj])
        toggled = LiftedAtom(predicates["toggled_on"], [soaker_obj])
        preconditions_toggleable = {reachable, holding_held, soakable, soaker, toggled}
        preconditions_not_toggleable = {reachable, holding_held, soakable, soaker, not_toggleable}
        add_effects = {LiftedAtom(predicates["soaked"], [soak_obj])}
        delete_effects = set()
        nsrt = NSRT(
            "Soak-toggleable", parameters,
            preconditions_toggleable, add_effects, delete_effects, set(),
            options["Soak"], option_vars, null_sampler
        )
        nsrts.add(nsrt)
        nsrt = NSRT(
            "Soak-notToggleable", parameters,
            preconditions_not_toggleable, add_effects, delete_effects, set(),
            options["Soak"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # CleanDusty
        clean_obj = Variable("?obj", object_type)
        held_obj = Variable("?held", object_type)
        parameters = [robot_obj, held_obj, clean_obj]
        option_vars = [robot_obj, clean_obj]
        reachable = LiftedAtom(predicates["reachable"], [clean_obj])
        holding_held = LiftedAtom(predicates["holding"], [held_obj])
        dusty = LiftedAtom(predicates["dusty"], [clean_obj])
        cleaner = LiftedAtom(predicates["cleaner"], [held_obj])
        preconditions = {reachable, holding_held, dusty, cleaner}
        add_effects = {LiftedAtom(predicates["not-dusty"], [clean_obj])}
        delete_effects = {dusty}
        nsrt = NSRT(
            "CleanDusty", parameters,
            preconditions, add_effects, delete_effects, set(),
            options["CleanDusty"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # CleanStained
        clean_obj = Variable("?obj", object_type)
        held_obj = Variable("?held", object_type)
        parameters = [robot_obj, held_obj, clean_obj]
        option_vars = [robot_obj, clean_obj]
        reachable = LiftedAtom(predicates["reachable"], [clean_obj])
        holding_held = LiftedAtom(predicates["holding"], [held_obj])
        stained = LiftedAtom(predicates["stained"], [clean_obj])
        cleaner = LiftedAtom(predicates["cleaner"], [held_obj])
        soaked = LiftedAtom(predicates["soaked"], [held_obj])
        preconditions = {reachable, holding_held, stained, cleaner, soaked}
        add_effects = {LiftedAtom(predicates["not-stained"], [clean_obj])}
        delete_effects = {stained}
        nsrt = NSRT(
            "CleanStained", parameters,
            preconditions, add_effects, delete_effects, set(),
            options["CleanStained"], option_vars, null_sampler
        )
        nsrts.add(nsrt)

        # Slice
        slice_obj = Variable("?obj", object_type)
        held_obj = Variable("?held", object_type)
        parameters = [robot_obj, held_obj, slice_obj]
        option_vars = [robot_obj, slice_obj]
        reachable = LiftedAtom(predicates["reachable"], [slice_obj])
        holding_held = LiftedAtom(predicates["holding"], [held_obj])
        sliceable = LiftedAtom(predicates["sliceable"], [slice_obj])
        slicer = LiftedAtom(predicates["slicer"], [held_obj])
        preconditions = {reachable, holding_held, sliceable, slicer}
        add_effects = {LiftedAtom(predicates["sliced"], [slice_obj])}
        delete_effects = set()
        nsrt = NSRT(
            "Slice", parameters,
            preconditions, add_effects, delete_effects, set(),
            options["Slice"], option_vars, null_sampler
        )
        nsrts.add(nsrt)


        return nsrts