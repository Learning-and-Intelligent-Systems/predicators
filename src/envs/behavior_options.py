"""Hardcoded options for BehaviorEnv.
"""
import argparse
import os
import parser
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt
import scipy

import bddl

from igibson import object_states
from igibson.envs.behavior_env import BehaviorEnv
from igibson.metrics.agent import BehaviorRobotMetric, FetchRobotMetric
from igibson.metrics.disarrangement import KinematicDisarrangement, LogicalDisarrangement
from igibson.metrics.task import TaskMetric
from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS
from igibson.objects.articulated_object import URDFObject
from igibson.utils.behavior_robot_planning_utils import plan_base_motion_br, plan_hand_motion_br
from igibson.utils import sampling_utils
from igibson.external.pybullet_tools.utils import (
    get_aabb,
    get_aabb_center,
    get_aabb_extent,
)

_ON_TOP_RAY_CASTING_SAMPLING_PARAMS = {
    # "hit_to_plane_threshold": 0.1,  # TODO: Tune this parameter.
    "max_angle_with_z_axis": 0.17,
    "bimodal_stdev_fraction": 1e-6,
    "bimodal_mean_fraction": 1.0,
    "max_sampling_attempts": 50,
    "aabb_offset": 0.01,
}

def get_body_ids(env, include_self=False):
    ids = []
    for object in env.scene.get_objects():
        if isinstance(object, URDFObject):
            # We want to exclude the floor since we're always floating and will never
            # practically collide with it, but if we include it in collision checking, we always
            # seem to collide.
            if object.name != 'floors': 
                ids.extend(object.body_ids)

    if include_self:
        ids.append(env.robots[0].parts["left_hand"].get_body_id())
        ids.append(env.robots[0].parts["body"].get_body_id())
        ids.append(env.robots[0].parts["eye"].get_body_id())
        ids.append(env.robots[0].parts["right_hand"].get_body_id())

    return ids

def detect_collision(bodyA, object_in_hand=None):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA or body_id == object_in_hand:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision

def detect_robot_collision(robot):
    object_in_hand = robot.parts["right_hand"].object_in_hand
    return (
        detect_collision(robot.parts["body"].body_id)
        or detect_collision(robot.parts["left_hand"].body_id)
        or detect_collision(robot.parts["right_hand"].body_id, object_in_hand)
    )

def get_aabb_volume(lo, hi):
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]

def get_closest_point_on_aabb(xyz, lo, hi):
    """Get the closest point on an aabb from a particular xyz coordinate"""
    closest_point_on_aabb = [0.0, 0.0, 0.0]
    for i in range(3):
        # if the coordinate is between the min and max of the aabb, then
        # use that coordinate directly
        if xyz[i] < hi[i] and xyz[i] > lo[i]:
            closest_point_on_aabb[i] = xyz[i]
        else:
            if abs(xyz[i] - hi[i]) < abs(xyz[i] - lo[i]):
                closest_point_on_aabb[i] = hi[i]
            else:
                closest_point_on_aabb[i] = lo[i]

    return closest_point_on_aabb

def reset_and_release_hand(env):
    env.robots[0].set_position_orientation(env.robots[0].get_position(), env.robots[0].get_orientation())
    for _ in range(100):
        env.robots[0].parts["right_hand"].set_close_fraction(0)
        env.robots[0].parts["right_hand"].trigger_fraction = 0
        p.stepSimulation()

def get_metrics_callbacks(config):
    metrics = [
        KinematicDisarrangement(),
        LogicalDisarrangement(),
        TaskMetric(),
    ]

    robot_type = config["robot"]
    if robot_type == "FetchGripper":
        metrics.append(FetchRobotMetric())
    elif robot_type == "BehaviorRobot":
        metrics.append(BehaviorRobotMetric())
    else:
        Exception("Metrics only implemented for FetchGripper and BehaviorRobot")

    return (
        [metric.start_callback for metric in metrics],
        [metric.step_callback for metric in metrics],
        [metric.end_callback for metric in metrics],
        [metric.gather_results for metric in metrics],
    )

def get_robot_pos(env):
    return np.array(env.robots[0].get_position())

def get_robot_eul(env):
    return np.array(p.getEulerFromQuaternion(env.robots[0].get_orientation()))

# Get low-level actions from base movement plan
def get_delta_low_level_base_action(env, original_orientation, old_xytheta, new_xytheta):
    
    ret_action = np.zeros(17)
    
    robot_z = env.robots[0].get_position()[2]

    # First, get the old and new position and orientation in the world frame as numpy arrays
    old_pos = np.array([old_xytheta[0], old_xytheta[1], robot_z])
    old_orn_quat = p.getQuaternionFromEuler(np.array([original_orientation[0], original_orientation[1], old_xytheta[2]]))
    new_pos = np.array([new_xytheta[0], new_xytheta[1], robot_z])
    new_orn_quat = p.getQuaternionFromEuler(np.array([original_orientation[0], original_orientation[1], new_xytheta[2]]))

    # Then, simply get the delta position and orientation by multiplying the inverse of the old pose by the 
    # new pose
    inverted_old_pos, inverted_old_orn_quat = p.invertTransform(old_pos, old_orn_quat)
    delta_pos, delta_orn_quat = p.multiplyTransforms(inverted_old_pos, inverted_old_orn_quat, new_pos, new_orn_quat)

    # Finally, convert the orientation back to euler angles from a quaternion
    delta_orn = p.getEulerFromQuaternion(delta_orn_quat)

    ret_action[0:3] = np.array([delta_pos[0], delta_pos[1], delta_orn[2]])

    return ret_action


#################

# Navigate To #

def navigate_to_param_sampler(rng):
    distance_to_try = [0.6, 1.2, 1.8, 2.4]
    distance = rng.choice(distance_to_try)
    yaw = rng.random() * (2 * np.pi) - np.pi
    return np.array([distance * np.cos(yaw), distance * np.sin(yaw)])

def navigate_to_obj_pos(env, obj, pos_offset, rng=np.random.default_rng(23)):
    """
    Parameterized controller for navigation.
    Runs motion planning to find a feasible trajectory to a certain x,y position offset from obj 
    and selects an orientation such that the robot is facing the object. If the navigation is infeasible,
    returns an indication to this effect.
    :param obj: an object to navigate toward
    :param to_pos: a length-2 numpy array (x, y) containing a position to navigate to
    :return: navigateToOption: a function that takes in a state and env (though the state is not used)
    and itself returns a low-level action and 'done' bit. Note that this return value may be None if
    no plan could be found.
    """
    # test agent positions around an obj
    # try to place the agent near the object, and rotate it to the object
    valid_position = None  # ((x,y,z),(roll, pitch, yaw))
    original_orientation = env.robots[0].get_orientation()
    state = p.saveState()

    def sample_fn(env, rng):
        random_point = env.scene.get_random_point(rng=rng)
        x, y = random_point[1][:2]
        theta = (rng.random() * (CIRCULAR_LIMITS[1] - CIRCULAR_LIMITS[0])) + CIRCULAR_LIMITS[0]
        return (x, y, theta)

    if isinstance(obj, URDFObject): # must be a URDFObject so we can get its position!
        obj_pos = obj.get_position()
        pos = [pos_offset[0] + obj_pos[0], pos_offset[1] + obj_pos[1], env.robots[0].initial_z_offset] 
        yaw_angle = np.arctan2(pos_offset[1], pos_offset[0]) - np.pi
        orn = [0, 0, yaw_angle]
        env.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
        eye_pos = env.robots[0].parts["eye"].get_position()
        ray_test_res = p.rayTest(eye_pos, obj_pos)
        # Test to see if the robot is obstructed by some object, but make sure that object 
        # is not either the robot's body or the object we want to pick up!
        blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (env.robots[0].parts["body"].get_body_id(), obj.get_body_id()))
        if not detect_robot_collision(env.robots[0]) and not blocked:
            valid_position = (pos, orn)
    else:
        print("ERROR! Object to navigate to is not valid (not an instance of URDFObject).")
        p.restoreState(state)
        p.removeState(state)
        print(f"PRIMITIVE: navigate to {obj.name} with params {pos_offset} fail")
        return None

    if valid_position is not None:
        p.restoreState(state)
        p.removeState(state)
        state = p.saveState()
        obstacles = get_body_ids(env)
        if env.robots[0].parts["right_hand"].object_in_hand is not None:
            obstacles.remove(env.robots[0].parts["right_hand"].object_in_hand)
        plan = plan_base_motion_br(
            robot=env.robots[0],
            end_conf=[valid_position[0][0], valid_position[0][1], valid_position[1][2]],
            base_limits=(),
            obstacles=obstacles,
            override_sample_fn= lambda: sample_fn(env, rng),
            rng=rng
        )

        p.restoreState(state)

        import ipdb; ipdb.set_trace()

        if plan is not None:

            def navigateToOption(state, env):

                atol_xy = 1e-2
                atol_theta = 1e-3
                atol_vel = 1e-4
                
                # 1. Get current position and orientation
                current_pos = list(env.robots[0].get_position()[0:2])
                current_orn = p.getEulerFromQuaternion(env.robots[0].get_orientation())[2]

                expected_pos = np.array(plan[0][0:2])
                expected_orn = np.array(plan[0][2])

                # 2. if error is greater that MAX_ERROR
                if not np.allclose(current_pos, expected_pos, atol=atol_xy) or not np.allclose(current_orn, expected_orn, atol=atol_theta):
                    # 2.a take a corrective action 
                    if len(plan) <= 1:
                        print("Plan of Length Zero")
                        done_bit = True
                        return np.zeros(17), done_bit
                    low_level_action = get_delta_low_level_base_action(env, original_orientation, np.array(current_pos + [current_orn]), np.array(plan[0]))
                
                    # But if the corrective action is 0
                    if np.allclose(low_level_action, np.zeros((17,1)), atol=atol_vel):
                        low_level_action = get_delta_low_level_base_action(env, original_orientation, np.array(current_pos + [current_orn]), np.array(plan[1]))
                        plan.pop(0)
                                                
                    return low_level_action, False

                else:
                    if len(plan) == 1: # In this case, we're at the final position we wanted to reach
                        low_level_action = np.zeros(17, dtype=float)
                        done_bit = True
                    
                    else:
                        low_level_action = get_delta_low_level_base_action(env, original_orientation, np.array(plan[0]), np.array(plan[1]))
                        done_bit = False
                    
                    plan.pop(0)

                    return low_level_action, done_bit

            p.restoreState(state)
            p.removeState(state)

            return navigateToOption

        else:
            p.restoreState(state)
            p.removeState(state)
            print("PRIMITIVE: navigate to {} failed; birrt failed to sample a plan!".format(obj.name))
            return None
        
    else:
        print("Position commanded is in collision or blocked!")
        p.restoreState(state)
        p.removeState(state)
        print(f"PRIMITIVE: navigate to {obj.name} with params {pos_offset} fail")
        return None


#################

# Grasp #

# Sampler for grasp continuous params
def grasp_obj_param_sampler(rng):
    x_offset = (rng.random() * 0.4) - 0.2
    y_offset = (rng.random() * 0.4) - 0.2
    z_offset = (rng.random() * 0.4) - 0.2
    z_rot = (rng.random() * 2 * np.pi) - np.pi
    return np.array([x_offset, y_offset, z_offset, z_rot])

# Get low level actions from hand-movement plan
def get_delta_low_level_hand_action(env, old_pos, old_orn, new_pos, new_orn):
    # First, convert the supplied orientations to quaternions
    old_orn = p.getQuaternionFromEuler(old_orn)
    new_orn = p.getQuaternionFromEuler(new_orn)

    # Next, find the inverted position of the body (which we know shouldn't change, since our actions 
    # move either the body or the hand, but not both simultaneously)
    body = env.robots[0].parts["right_hand"].parent.parts["body"]
    inverted_body_new_pos, inverted_body_new_orn = p.invertTransform(body.new_pos, body.new_orn)
    # Use this to compute the new pose of the hand w.r.t the body frame
    new_local_pos, new_local_orn = p.multiplyTransforms(inverted_body_new_pos, inverted_body_new_orn, new_pos, new_orn)
    
    # Next, compute the old pose of the hand w.r.t the body frame 
    inverted_body_old_pos = inverted_body_new_pos
    inverted_body_old_orn = inverted_body_new_orn
    old_local_pos, old_local_orn = p.multiplyTransforms(inverted_body_old_pos, inverted_body_old_orn, old_pos, old_orn)
    
    # The delta position is simply given by the difference between these positions
    delta_pos = np.array(new_local_pos) - np.array(old_local_pos)

    # Finally, compute the delta orientation
    inverted_old_local_orn_pos, inverted_old_local_orn_orn = p.invertTransform([0, 0, 0], old_local_orn)
    _, delta_orn = p.multiplyTransforms([0, 0, 0], new_local_orn, inverted_old_local_orn_pos, inverted_old_local_orn_orn)

    delta_trig_frac = 0
    action = np.concatenate([np.zeros((10)), np.array(delta_pos), np.array(p.getEulerFromQuaternion(delta_orn)), np.array([delta_trig_frac])], axis=0)

    return action

def grasp_obj_at_pos(env, obj, grasp_offset_and_z_rot, rng=np.random.default_rng(23)):
    plan = np.zeros((17,1))
    if env.obj_in_hand is None:
        if isinstance(obj, URDFObject) and hasattr(obj, "states") and object_states.AABB in obj.states:
            lo, hi = obj.states[object_states.AABB].get_value()
            volume = get_aabb_volume(lo, hi)
            if (
                volume < 0.3 * 0.3 * 0.3 and not obj.main_body_is_fixed
            ):  # say we can only grasp small objects
                if (
                    np.linalg.norm(np.array(obj.get_position()) - np.array(env.robots[0].get_position()))
                    < 2
                ):

                    ### Grasping Phase 1: Compute the position and orientation of the hand based on the 
                    # provided continuous parameters and try to create a plan to it.
                    obj_pos = obj.get_position()
                    x = obj_pos[0] + grasp_offset_and_z_rot[0]
                    y = obj_pos[1] + grasp_offset_and_z_rot[1]
                    z = obj_pos[2] + grasp_offset_and_z_rot[2]
                    hand_x, hand_y, hand_z = env.robots[0].parts["right_hand"].get_position()

                    # # add a little randomness to avoid getting stuck
                    # x += np.random.uniform(-0.025, 0.025)
                    # y += np.random.uniform(-0.025, 0.025)
                    # z += np.random.uniform(-0.025, 0.025)

                    minx = min(x, hand_x) - 0.5
                    miny = min(y, hand_y) - 0.5
                    minz = min(z, hand_z) - 0.5
                    maxx = max(x, hand_x) + 0.5
                    maxy = max(y, hand_y) + 0.5
                    maxz = max(z, hand_z) + 0.5

                    # compute the angle the hand must be in such that it can grasp the object from its current offset position
                    # This involves aligning the z-axis (in the world frame) of the hand with the vector that goes from the hand 
                    # to the object. We can find the rotation matrix that accomplishes this rotation by following:
                    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
                    hand_to_obj_vector = np.array([x - hand_x, y - hand_y, z - hand_z])
                    hand_to_obj_unit_vector = hand_to_obj_vector / np.linalg.norm(hand_to_obj_vector)
                    unit_z_vector = np.array([0,0,1])
                    c = np.dot(unit_z_vector, hand_to_obj_unit_vector)
                    if not c == -1.0:
                        v = np.cross(unit_z_vector, hand_to_obj_unit_vector)
                        s = np.linalg.norm(v)
                        v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        R = np.eye(3) + v_x + np.linalg.matrix_power(v_x, 2) * ((1-c)/(s ** 2))
                        r = scipy.spatial.transform.Rotation.from_matrix(R)
                        euler_angles = r.as_euler('xyz')
                        euler_angles[2] += grasp_offset_and_z_rot[3]
                    else:
                        euler_angles = np.array([0.0, np.pi, 0.0])

                    state = p.saveState()
                    # plan a motion to the pose [x, y, z, euler_angles[0], euler_angles[1], euler_angles[2]]
                    plan = plan_hand_motion_br(
                        robot=env.robots[0],
                        obj_in_hand=None,
                        end_conf=[x, y, z, euler_angles[0], euler_angles[1], euler_angles[2]],
                        hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
                        obstacles=get_body_ids(env, include_self=False),
                        rng=rng
                    )
                    p.restoreState(state)

                    # NOTE: This below line is *VERY* important after the pybullet state is restored. The hands keep an internal track of their state, and if we don't reset their internal state to mirror the 
                    # actual pybullet state, the hand will think its elsewhere and update incorrectly accordingly
                    env.robots[0].parts["right_hand"].set_position(env.robots[0].parts["right_hand"].get_position())
                    env.robots[0].parts["left_hand"].set_position(env.robots[0].parts["left_hand"].get_position())

                    ### Grasping Phase 2: Move along the vector from the position the hand ends up in to the 
                    # object and then try to grasp.

                    if plan is not None:
                        hand_pos = plan[-1][0:3]
                        hand_orn = plan[-1][3:6]
                        # Get the closest point on the object's bounding box at which we can try to put the hand
                        closest_point_on_aabb = get_closest_point_on_aabb(hand_pos, lo, hi)

                        delta_pos_to_obj = [closest_point_on_aabb[0] - hand_pos[0], closest_point_on_aabb[1] - hand_pos[1], closest_point_on_aabb[2] - hand_pos[2]]
                        delta_step_to_obj = [delta_pos / 25.0 for delta_pos in delta_pos_to_obj] # because we want to accomplish the motion in 25 timesteps

                        # lower the hand until it touches the object
                        for _ in range(25):
                            new_hand_pos = [hand_pos[0] + delta_step_to_obj[0], hand_pos[1] + delta_step_to_obj[1], hand_pos[2] + delta_step_to_obj[2]]
                            plan.append(new_hand_pos + list(hand_orn))
                            hand_pos = new_hand_pos

                        # Setup two booleans to be used as 'memory', as well as a 'reversed' plan to be used 
                        # for our option that's defined below
                        reversed_plan = [rev_elem for rev_elem in reversed(plan[:])]
                        plan_executed_forwards = False
                        tried_closing_gripper = False

                        # TODO: Include the error-correcting closed-loop execution actions in here.
                        def graspObjectOption(state, env):
                            nonlocal plan_executed_forwards
                            nonlocal tried_closing_gripper
                            done_bit = False
                            if not plan_executed_forwards and not tried_closing_gripper:
                                # Step thru the plan to execute Grasping phases 1 and 2
                                ret_action = get_delta_low_level_hand_action(env, plan[0][0:3], plan[0][3:6], plan[1][0:3], plan[1][3:6])
                                plan.pop(0)
                                if len(plan) == 1:
                                    plan_executed_forwards = True
                                
                            elif plan_executed_forwards and not tried_closing_gripper:
                                # Close the gripper to see if you've gotten the object
                                ret_action = np.zeros(17, dtype=float)
                                ret_action[16] = 1.0
                                tried_closing_gripper = True

                            else:
                                # Grasping Phase 3: getting the hand back to resting position
                                # near the robot.
                                ret_action = get_delta_low_level_hand_action(env, reversed_plan[0][0:3], reversed_plan[0][3:6], reversed_plan[1][0:3], reversed_plan[1][3:6])
                                reversed_plan.pop(0)
                                if len(reversed_plan) == 1:
                                    done_bit = True

                            return ret_action, done_bit

                        p.restoreState(state)
                        p.removeState(state)

                        return graspObjectOption

                    else:
                        print(f"PRIMITIVE: grasp {obj.name} fail, failed to find plan to continuous params {grasp_offset_and_z_rot}")
                        return None

                else:
                    print("PRIMITIVE: grasp {} fail, too far".format(obj.name))
                    return None
            else:
                print("PRIMITIVE: grasp {} fail, too big or fixed".format(obj.name))
                return None
        else:
            print("PRIMITIVE: grasp {} fail, no object".format(obj.name))
            return None
    else:
        print("PRIMITIVE: grasp {} fail, agent already has an object in hand!".format(obj.name))
        return None

#################

#################

# Place Ontop #

def place_obj_plan(env, obj, original_state, place_rel_pos, rng=np.random.default_rng(23)):
    obj_in_hand = env.scene.get_objects()[env.robots[0].parts["right_hand"].object_in_hand]
    x, y, z = np.add(place_rel_pos, obj.get_position())
    hand_x, hand_y, hand_z = env.robots[0].parts["right_hand"].get_position()

    minx = min(x, hand_x) - 1
    miny = min(y, hand_y) - 1
    minz = min(z, hand_z) - 0.5
    maxx = max(x, hand_x) + 1
    maxy = max(y, hand_y) + 1
    maxz = max(z, hand_z) + 0.5

    obstacles = get_body_ids(env, include_self=False)
    obstacles.remove(env.robots[0].parts["right_hand"].object_in_hand)
    plan = plan_hand_motion_br(
        robot=env.robots[0],
        obj_in_hand=obj_in_hand,
        end_conf=[x, y, z + 0.2, 0, np.pi * 7/6, 0], #TODO FIX to make HAND Face Table (Can Sample These)
        hand_limits=((minx, miny, minz), (maxx, maxy, maxz)),
        obstacles=obstacles,
        rng=rng
    )  #
    p.restoreState(original_state)
    p.removeState(original_state)

    # NOTE: This below line is *VERY* important after the pybullet state is restored. The hands keep an internal track of their state, and if we don't reset their internal state to mirror the 
    # actual pybullet state, the hand will think its elsewhere and update incorrectly accordingly
    env.robots[0].parts["right_hand"].set_position(env.robots[0].parts["right_hand"].get_position())
    env.robots[0].parts["left_hand"].set_position(env.robots[0].parts["left_hand"].get_position())

    return plan

def place_ontop_obj_pos_sampler(env, obj, return_orn=False, rng=np.random.default_rng(23)):
    # Sample Params #
    objA = env.scene.get_objects()[env.robots[0].parts["right_hand"].object_in_hand]
    objB = obj
    params = _ON_TOP_RAY_CASTING_SAMPLING_PARAMS
    aabb = get_aabb(objA.get_body_id())
    aabb_center, aabb_extent = get_aabb_center(aabb), get_aabb_extent(aabb)

    sampling_results = sampling_utils.sample_cuboid_on_object( # Non rng???
        objB,
        num_samples=1,
        cuboid_dimensions=aabb_extent,
        axis_probabilities=[0, 0, 1],
        refuse_downwards=True,
        **params
        )

    if sampling_results[0] is None:
        return None

    rnd_params = np.subtract(sampling_results[0][0], obj.get_position())

    if return_orn:
        return rnd_params, sampling_results[0][2]

    return rnd_params


def place_ontop_obj_pos(env, obj, place_rel_pos, place_orn=None, option_model=False, rng=np.random.default_rng(23)):
    plan = np.zeros((17,1))
    obj_in_hand = env.scene.get_objects()[env.robots[0].parts["right_hand"].object_in_hand]
    if obj_in_hand is not None and obj_in_hand != obj:
        print("PRIMITIVE:attempt to place {} ontop {}".format(obj_in_hand.name, obj.name))

        if isinstance(obj, URDFObject):
            if np.linalg.norm(np.array(obj.get_position()) - np.array(env.robots[0].get_position())) < 2:
                original_orientation = env.robots[0].get_orientation()
                state = p.saveState()

                # To check if object fits on place location
                p.restoreState(state)

                # NOTE: This below line is *VERY* important after the pybullet state is restored. The hands keep an internal track of their state, and if we don't reset their internal state to mirror the 
                # actual pybullet state, the hand will think its elsewhere and update incorrectly accordingly
                env.robots[0].parts["right_hand"].set_position(env.robots[0].parts["right_hand"].get_position())
                env.robots[0].parts["left_hand"].set_position(env.robots[0].parts["left_hand"].get_position())

                
                if not option_model:
                    plan = place_obj_plan(env, obj, state, place_rel_pos, rng=rng)
                    if plan is None:
                        return None
                    reversed_plan = [rev_elem for rev_elem in reversed(plan[:])]
                    plan_executed_forwards = False
                    tried_opening_gripper = False

                    print(
                        "PRIMITIVE: place {} ontop {} success".format(obj_in_hand.name, obj.name)
                    )

                    # TODO: Include the error-correcting closed-loop execution actions in here.
                    def placeOntopObjectOption(state, env):
                        nonlocal plan
                        nonlocal plan_executed_forwards
                        nonlocal tried_opening_gripper

                        done_bit = False

                        atol_xy = 0.1
                        atol_theta = 0.1
                        atol_vel = 0.5
                        
                        # 1. Get current position and orientation
                        current_pos, current_orn_quat = p.multiplyTransforms(env.robots[0].parts["right_hand"].parent.parts["body"].new_pos, env.robots[0].parts["right_hand"].parent.parts["body"].new_orn, env.robots[0].parts["right_hand"].local_pos, env.robots[0].parts["right_hand"].local_orn)
                        current_orn = p.getEulerFromQuaternion(current_orn_quat)

                        expected_pos = np.array(plan[0][0:3])
                        expected_orn = np.array(plan[0][3:])

                        if not plan_executed_forwards and not tried_opening_gripper:
                            ###
                            # 2. if error is greater that MAX_ERROR
                            if not np.allclose(current_pos, expected_pos, atol=atol_xy) or not np.allclose(current_orn, expected_orn, atol=atol_theta):
                                # 2.a take a corrective action 
                                if len(plan) <= 1:
                                    print("Plan of Length Zero")
                                    done_bit = False
                                    plan_executed_forwards = True
                                    low_level_action = np.zeros(17)
                                    low_level_action[16] = 1.0
                                    #print("lla:", low_level_action)
                                    return low_level_action, done_bit
                                low_level_action = get_delta_low_level_hand_action(env, np.array(current_pos), np.array(current_orn), np.array(plan[0][0:3]), np.array(plan[0][3:]))

                                # But if the corrective action is 0
                                if np.allclose(low_level_action, np.zeros((17,1)), atol=atol_vel):
                                    low_level_action = get_delta_low_level_hand_action(env, np.array(current_pos), np.array(current_orn), np.array(plan[1][0:3]), np.array(plan[1][3:]))
                                    plan.pop(0)
                                low_level_action[16] = 1.0  
                                #print("lla:", low_level_action)             
                                return low_level_action, False

                            else:
                                if len(plan) <= 1: # In this case, we're at the final position we wanted to reach
                                    low_level_action = np.zeros(17, dtype=float)
                                    done_bit = False
                                    plan_executed_forwards = True
                                
                                else:
                                    # Step thru the plan to execute Grasping phases 1 and 2
                                    low_level_action = get_delta_low_level_hand_action(env, plan[0][0:3], plan[0][3:], plan[1][0:3], plan[1][3:])
                                    low_level_action[16] = 1.0
                                    if len(plan) == 1:
                                        plan_executed_forwards = True
                                
                                plan.pop(0)
                                low_level_action[16] = 1.0
                                #print("lla:", low_level_action)
                                return low_level_action, done_bit

                            ###
                            
                        elif plan_executed_forwards and not tried_opening_gripper:
                            # Close the gripper to see if you've gotten the object
                            low_level_action = np.zeros(17, dtype=float)
                            low_level_action[16] = 0.0
                            tried_opening_gripper = True
                            return low_level_action, False

                        else:
                            plan = reversed_plan
                            ###
                            # 2. if error is greater that MAX_ERROR
                            if not np.allclose(current_pos, expected_pos, atol=atol_xy) or not np.allclose(current_orn, expected_orn, atol=atol_theta):
                                # 2.a take a corrective action 
                                if len(plan) <= 1:
                                    print("Plan of Length Zero")
                                    done_bit = True
                                    return np.zeros(17), done_bit
                                low_level_action = get_delta_low_level_hand_action(env, np.array(current_pos), np.array(current_orn), np.array(plan[0][0:3]), np.array(plan[0][3:]))
                            
                                # But if the corrective action is 0
                                if np.allclose(low_level_action, np.zeros((17,1)), atol=atol_vel):
                                    low_level_action = get_delta_low_level_hand_action(env, np.array(current_pos), np.array(current_orn), np.array(plan[1][0:3]), np.array(plan[1][3:]))
                                    plan.pop(0)
                                                            
                                return low_level_action, False

                            else:
                                if len(plan) == 1: # In this case, we're at the final position we wanted to reach
                                    low_level_action = np.zeros(17, dtype=float)
                                    done_bit = True
                                
                                else:
                                    # Grasping Phase 3: getting the hand back to resting position
                                    # near the robot.
                                    low_level_action = get_delta_low_level_hand_action(env, reversed_plan[0][0:3], reversed_plan[0][3:], reversed_plan[1][0:3], reversed_plan[1][3:])
                                    low_level_action[16] = 1.0
                                    if len(reversed_plan) == 1:
                                        done_bit = True
                                
                                reversed_plan.pop(0)

                                return low_level_action, done_bit

                            ###
                else:
                    def placeOntopObjectOptionModel(init_state, env):
                        target_pos = place_rel_pos
                        target_orn = place_orn
                        env.robots[0].parts["right_hand"].force_release_obj()
                        obj_in_hand.set_position_orientation(target_pos, target_orn)

                        final_state = env.get_state()
                        return final_state, True

                if option_model:
                    return placeOntopObjectOptionModel
                else:
                    return placeOntopObjectOption

            else:
                print(
                    "PRIMITIVE: place {} ontop {} fail, too far".format(obj_in_hand.name, obj.name)
                )
                return None


#################