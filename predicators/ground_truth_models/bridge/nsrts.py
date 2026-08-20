"""Ground-truth NSRTs for the bridge (glue construction) environment.

Manipulation NSRTs (pick/place/stack/butt-join/seat/glue/wait) mirror
the endogenous processes. There is deliberately no NSRT that adds
``Attached`` -- curing is a delayed exogenous process, so the
environment is solved/demoed by ``oracle_process_planning`` (like
``pybullet_boil`` and ``pybullet_bond``), not plain sesame ``oracle``.
"""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.ground_truth_models.bridge.processes import \
    _apply_glue_sampler, _pick_sampler, _place_block_on_table_sampler, \
    _place_bottle_sampler, _place_leg_at_site_sampler, \
    _place_next_to_sampler, _seat_span_sampler, _stack_leg_sampler
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletBridgeGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the bridge environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_bridge"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        del env_name
        robot_type = types["robot"]
        block_type = types["block"]
        bottle_type = types["bottle"]
        site_type = types["site"]

        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        HoldingBottle = predicates["HoldingBottle"]
        GlueTop = predicates["GlueTop"]
        GlueEndB = predicates["GlueEndB"]
        OnBlock = predicates["OnBlock"]
        NextToEnd = predicates["NextToEnd"]
        SeatedOn = predicates["SeatedOn"]
        AtSite = predicates["AtSite"]
        SiteFree = predicates["SiteFree"]
        Attached = predicates["Attached"]
        Standing = predicates["Standing"]
        Lying = predicates["Lying"]
        Loose = predicates["Loose"]
        Resting = predicates["Resting"]
        TopFree = predicates["TopFree"]

        PickBlock = options["PickBlock"]
        PickBottle = options["PickBottle"]
        Place = options["Place"]
        ApplyGlueTop = options["ApplyGlueTop"]
        ApplyGlueEndB = options["ApplyGlueEndB"]
        Wait = options["Wait"]

        nsrts: Set[NSRT] = set()

        # PickBlockFromTable
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        nsrts.add(
            NSRT("PickBlockFromTable", [robot, blk], {
                LiftedAtom(HandEmpty, [robot]),
                LiftedAtom(TopFree, [blk]),
            }, {LiftedAtom(Holding, [robot, blk])}, {
                LiftedAtom(HandEmpty, [robot]),
                LiftedAtom(Resting, [blk]),
            }, set(), PickBlock, [robot, blk], _pick_sampler))

        # PickGlueBottle
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        nsrts.add(
            NSRT("PickGlueBottle", [robot, bottle],
                 {LiftedAtom(HandEmpty, [robot])},
                 {LiftedAtom(HoldingBottle, [robot, bottle])},
                 {LiftedAtom(HandEmpty, [robot])}, set(), PickBottle,
                 [robot, bottle], _pick_sampler))

        # PlaceBottleOnTable
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        nsrts.add(
            NSRT("PlaceBottleOnTable", [robot, bottle],
                 {LiftedAtom(HoldingBottle, [robot, bottle])},
                 {LiftedAtom(HandEmpty, [robot])},
                 {LiftedAtom(HoldingBottle, [robot, bottle])}, set(), Place,
                 [robot], _place_bottle_sampler))

        # PlaceLegAtSite
        robot = Variable("?robot", robot_type)
        leg = Variable("?leg", block_type)
        site = Variable("?site", site_type)
        nsrts.add(
            NSRT(
                "PlaceLegAtSite", [robot, leg, site], {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                    LiftedAtom(Standing, [leg]),
                    LiftedAtom(Loose, [leg]),
                }, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(AtSite, [leg, site]),
                    LiftedAtom(Resting, [leg]),
                }, {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                }, set(), Place, [robot], _place_leg_at_site_sampler))

        # StackLegOnLeg
        robot = Variable("?robot", robot_type)
        top = Variable("?top", block_type)
        bottom = Variable("?bottom", block_type)
        site = Variable("?site", site_type)
        nsrts.add(
            NSRT(
                "StackLegOnLeg", [robot, top, bottom, site], {
                    LiftedAtom(Holding, [robot, top]),
                    LiftedAtom(AtSite, [bottom, site]),
                    LiftedAtom(GlueTop, [bottom]),
                    LiftedAtom(Standing, [top]),
                    LiftedAtom(Loose, [top]),
                }, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(OnBlock, [top, bottom]),
                    LiftedAtom(Resting, [top]),
                }, {
                    LiftedAtom(Holding, [robot, top]),
                    LiftedAtom(TopFree, [bottom]),
                }, set(), Place, [robot], _stack_leg_sampler))

        # PlaceSpanNextTo
        robot = Variable("?robot", robot_type)
        right = Variable("?right", block_type)
        left = Variable("?left", block_type)
        nsrts.add(
            NSRT(
                "PlaceSpanNextTo", [robot, right, left], {
                    LiftedAtom(Holding, [robot, right]),
                    LiftedAtom(GlueEndB, [left]),
                    LiftedAtom(Lying, [right]),
                    LiftedAtom(Lying, [left]),
                    LiftedAtom(Loose, [right]),
                }, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(NextToEnd, [right, left]),
                    LiftedAtom(Resting, [right]),
                }, {LiftedAtom(Holding, [robot, right])}, set(), Place,
                [robot], _place_next_to_sampler))

        # PlaceBlockOnTable
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        nsrts.add(
            NSRT("PlaceBlockOnTable", [robot, blk],
                 {LiftedAtom(Holding, [robot, blk])}, {
                     LiftedAtom(HandEmpty, [robot]),
                     LiftedAtom(Resting, [blk]),
                 }, {LiftedAtom(Holding, [robot, blk])}, set(), Place, [robot],
                 _place_block_on_table_sampler))

        # ApplyGlueTop / ApplyGlueEndB
        for glue_pred, shape_pred, option, name in ((GlueTop, Standing,
                                                     ApplyGlueTop,
                                                     "ApplyGlueTop"),
                                                    (GlueEndB, Lying,
                                                     ApplyGlueEndB,
                                                     "ApplyGlueEndB")):
            robot = Variable("?robot", robot_type)
            bottle = Variable("?bottle", bottle_type)
            blk = Variable("?block", block_type)
            conds = {
                LiftedAtom(HoldingBottle, [robot, bottle]),
                LiftedAtom(shape_pred, [blk]),
            }
            if glue_pred is GlueTop:
                conds.add(LiftedAtom(TopFree, [blk]))
            nsrts.add(
                NSRT(name, [robot, bottle, blk], conds,
                     {LiftedAtom(glue_pred, [blk])}, set(), set(), option,
                     [robot, bottle, blk], _apply_glue_sampler))

        # SeatSpan2 / SeatSpan3 (see processes.py for the condition
        # rationale -- these mirror the endogenous processes exactly).
        for n_span in (2, 3):
            robot = Variable("?robot", robot_type)
            span_a = Variable("?spanA", block_type)
            span_b = Variable("?spanB", block_type)
            leg_l = Variable("?legL", block_type)
            leg_r = Variable("?legR", block_type)
            if n_span == 2:
                span_vars = [span_a, span_b]
                chain = {LiftedAtom(Attached, [span_a, span_b])}
                site_l = Variable("?siteL", site_type)
                site_r = Variable("?siteR", site_type)
                leg_vars = [leg_l, leg_r, site_l, site_r]
                legs_ready = {
                    LiftedAtom(AtSite, [leg_l, site_l]),
                    LiftedAtom(AtSite, [leg_r, site_r]),
                }
            else:
                mid = Variable("?spanMid", block_type)
                span_vars = [span_a, mid, span_b]
                chain = {
                    LiftedAtom(Attached, [span_a, mid]),
                    LiftedAtom(Attached, [mid, span_b]),
                    LiftedAtom(Lying, [mid]),
                }
                base_l = Variable("?baseL", block_type)
                base_r = Variable("?baseR", block_type)
                leg_vars = [leg_l, leg_r, base_l, base_r]
                legs_ready = {
                    LiftedAtom(OnBlock, [leg_l, base_l]),
                    LiftedAtom(OnBlock, [leg_r, base_r]),
                }
            nsrts.add(
                NSRT(
                    f"SeatSpan{n_span}", [robot] + span_vars + leg_vars, {
                        LiftedAtom(Holding, [robot, span_a]),
                        LiftedAtom(GlueTop, [leg_l]),
                        LiftedAtom(GlueTop, [leg_r]),
                        LiftedAtom(Lying, [span_a]),
                        LiftedAtom(Lying, [span_b]),
                        LiftedAtom(Standing, [leg_l]),
                        LiftedAtom(Standing, [leg_r]),
                    } | chain | legs_ready, {
                        LiftedAtom(HandEmpty, [robot]),
                        LiftedAtom(SeatedOn, [span_a, leg_l]),
                        LiftedAtom(SeatedOn, [span_b, leg_r]),
                        LiftedAtom(Resting, [span_a]),
                    }, {
                        LiftedAtom(Holding, [robot, span_a]),
                        LiftedAtom(TopFree, [leg_l]),
                        LiftedAtom(TopFree, [leg_r]),
                    }, set(), Place, [robot], _seat_span_sampler))

        # Wait
        robot = Variable("?robot", robot_type)
        nsrts.add(
            NSRT("Wait", [robot], set(), set(), set(), set(), Wait, [robot],
                 null_sampler))

        return nsrts
