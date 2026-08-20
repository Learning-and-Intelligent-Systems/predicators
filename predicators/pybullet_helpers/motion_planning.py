"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence

import numpy as np
import pybullet as p
from gym.spaces import Box
from numpy.typing import NDArray

from predicators import utils
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: Optional[int] = None,
    base_link_to_held_obj: Optional[NDArray] = None,
    allow_shallow_held_object_contacts: bool = False,
    goal_finger_joint: Optional[float] = None,
    held_bystander_clearance: Optional[float] = None,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Collision bodies near the robot or held object at the start or goal
    configuration are treated as intended contact partners and checked
    against ``CFG.pybullet_birrt_contact_margin``; all other bodies are
    bystanders from which the path must keep
    ``CFG.pybullet_birrt_bystander_clearance`` of separation.

    When ``goal_finger_joint`` is given, the goal configuration is
    additionally checked with both finger joints at that value (e.g. a
    place phase whose next phase opens the gripper: the opening sweep is
    not otherwise planned, so a goal whose open fingers would hit a
    neighbor must be rejected here).

    ``held_bystander_clearance`` overrides
    ``CFG.pybullet_birrt_held_bystander_clearance`` (the wider berth the
    held object keeps from bodies the path never intends to approach).

    Note that this function changes the state of the robot.
    """
    rng = np.random.default_rng(seed)
    # BiRRT plans in the arm-joint space. For mobile robots, action_space also
    # includes base-delta dims (appended last); strip them so sampled configs
    # match the arm joints that set_joints / forward_kinematics expect. For
    # fixed-base robots (base_action_dim == 0) this is a no-op.
    joint_space = robot.action_space
    base_dim = int(getattr(robot, "base_action_dim", 0))
    if base_dim > 0:
        joint_space = Box(low=np.asarray(joint_space.low[:-base_dim]),
                          high=np.asarray(joint_space.high[:-base_dim]),
                          dtype=np.float32)
    joint_space.seed(seed)
    num_interp = CFG.pybullet_birrt_extend_num_interp

    def _sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
        new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        if held_object is not None:
            assert base_link_to_held_obj is not None
            world_to_base_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=physics_client_id).com_pose
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     base_link_to_held_obj[0],
                                                     base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                held_object,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=physics_client_id)

    hard_margin = CFG.pybullet_birrt_contact_margin
    shallow_margin = CFG.pybullet_birrt_shallow_held_contact_margin
    bystander_clearance = CFG.pybullet_birrt_bystander_clearance

    allowed_shallow_held_collision_bodies = set()
    if allow_shallow_held_object_contacts and held_object is not None:
        _set_state(initial_positions)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_bodies:
            contacts = p.getContactPoints(held_object,
                                          body,
                                          physicsClientId=physics_client_id)
            penetrating = [c[8] for c in contacts if c[8] < hard_margin]
            if penetrating and min(penetrating) >= shallow_margin:
                allowed_shallow_held_collision_bodies.add(body)

    # Bodies the robot or held object starts or deliberately ends within
    # the clearance of are intended contact partners (support surfaces,
    # grasp targets, placement neighbors) and keep the hard margin;
    # every other body is a bystander from which the whole path must
    # keep ``bystander_clearance`` of separation, so that a "collision-
    # free" path cannot physically graze it (the hard margin tolerates
    # ~1mm of penetration, enough to topple a knife-edge object).
    #
    # The held object additionally gets the larger
    # ``pybullet_birrt_held_bystander_clearance`` against bodies the path
    # has no business approaching at all: unlike the position-controlled
    # robot links, the held object hangs on a grasp constraint and lags
    # the end effector's mid-path orientation swings by ~0.05 rad
    # (centimetres at the tip of a long object), so a plan that clears a
    # bystander by only ``bystander_clearance`` still physically grazes
    # it. Bodies already within the held clearance at the start or goal
    # keep the plain ``bystander_clearance`` so deliberately tight
    # placements (butt joints, chain neighbors) stay plannable.
    held_clearance = held_bystander_clearance \
        if held_bystander_clearance is not None \
        else CFG.pybullet_birrt_held_bystander_clearance
    held_body_clearances: dict = {}
    contact_partners: set = set(collision_bodies)
    if bystander_clearance > hard_margin:
        contact_partners = set()
        held_probe_radius = max(bystander_clearance, held_clearance)
        held_near_endpoint: set = set()
        for pt in (initial_positions, target_positions):
            _set_state(pt)
            for body in collision_bodies:
                if body not in contact_partners:
                    if p.getClosestPoints(robot.robot_id,
                                          body,
                                          bystander_clearance,
                                          physicsClientId=physics_client_id):
                        contact_partners.add(body)
                if held_object is None or body in held_near_endpoint:
                    continue
                held_pts = p.getClosestPoints(
                    held_object,
                    body,
                    held_probe_radius,
                    physicsClientId=physics_client_id)
                if held_pts:
                    if min(pt[8] for pt in held_pts) < bystander_clearance:
                        contact_partners.add(body)
                    held_near_endpoint.add(body)
        if held_object is not None and held_clearance > bystander_clearance:
            held_body_clearances = {
                body: held_clearance
                for body in collision_bodies if body not in held_near_endpoint
            }

    def _extend_fn(pt1: JointPositions,
                   pt2: JointPositions) -> Iterator[JointPositions]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointPositions) -> bool:
        _set_state(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        # Use a penetration margin to distinguish real collisions from
        # resting-on-surface contacts (e.g. a grasped object sitting on a
        # table, or a gripper touching a just-released object).
        # getContactPoints returns tuples where index 8 is contactDistance:
        # negative = penetration, ~0 = touching, positive = separation.
        # Contact partners fail only on penetration deeper than the hard
        # margin; bystanders additionally fail on separations below the
        # clearance (Bullet generates contact points out to its
        # contactBreakingThreshold, 0.02 by default, so millimetre-scale
        # positive distances are reported here).
        for body in collision_bodies:
            margin = hard_margin if body in contact_partners \
                else bystander_clearance
            contacts = p.getContactPoints(robot.robot_id,
                                          body,
                                          physicsClientId=physics_client_id)
            if any(c[8] < margin for c in contacts):
                return True
            if held_object is not None:
                # Clearances above Bullet's contactBreakingThreshold
                # (0.02 m) would be silently unenforced: getContactPoints
                # generates no points beyond it. Query closest points out
                # to the clearance instead when the held clearance
                # applies.
                held_margin = held_body_clearances.get(body, margin)
                contacts = p.getClosestPoints(
                    held_object,
                    body,
                    held_margin,
                    physicsClientId=physics_client_id) \
                    if held_margin > 0 else p.getContactPoints(
                        held_object, body, physicsClientId=physics_client_id)
                contact_distances = [c[8] for c in contacts]
                if body in allowed_shallow_held_collision_bodies:
                    if any(d < shallow_margin for d in contact_distances):
                        return True
                    continue
                if any(d < held_margin for d in contact_distances):
                    return True
        return False

    if goal_finger_joint is not None:
        release_config = list(target_positions)
        release_config[robot.left_finger_joint_idx] = goal_finger_joint
        release_config[robot.right_finger_joint_idx] = goal_finger_joint
        if _collision_fn(release_config):
            return None

    def _distance_fn(from_pt: JointPositions, to_pt: JointPositions) -> float:
        # NOTE: only using positions to calculate distance. Should use
        # orientations as well in the near future.
        from_ee = robot.forward_kinematics(from_pt).position
        to_ee = robot.forward_kinematics(to_pt).position
        return sum(np.subtract(from_ee, to_ee)**2)

    birrt = utils.BiRRT(_sample_fn,
                        _extend_fn,
                        _collision_fn,
                        _distance_fn,
                        rng,
                        num_attempts=CFG.pybullet_birrt_num_attempts,
                        num_iters=CFG.pybullet_birrt_num_iters,
                        smooth_amt=CFG.pybullet_birrt_smooth_amt)

    path = birrt.query(initial_positions, target_positions)
    if path is not None and CFG.pybullet_birrt_path_subsample_ratio > 1:
        ratio = CFG.pybullet_birrt_path_subsample_ratio
        last = path[-1]
        path = [path[i] for i in range(0, len(path), ratio)]
        # Always include the final waypoint.
        if path[-1] is not last:
            path.append(last)
    return path
