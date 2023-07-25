"""Testing the feasibility of motion planning in kitchen environment."""

from typing import Iterator

import numpy as np
from pyquaternion import Quaternion
import mujoco_py
# from gym_kuka_mujoco.utils.kinematics import inverseKin

from predicators import utils
from predicators.envs import create_new_env
from predicators.structs import Array


DOWN_QUAT = np.array([0.0, 0.707, 0.707, 0.0])

def _main() -> None:
    utils.reset_config({
        "env": "kitchen",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
    })
    use_gui = True
    env = create_new_env("kitchen", use_gui=use_gui)
    gym_env = env._gym_env
    init_obs = env.reset("train", 0)
    target_pose = np.add(init_obs["state_info"]["knob3_site"], [0.0, -0.1, 0.0])
    target_quat = Quaternion(DOWN_QUAT)
    target_pose_eps = 100.0 #0.01
    target_quat_eps = 0.1
    start = gym_env.get_robot_joint_position()
    num_dof = 7

    cspace = gym_env.robot.robot_pos_bound[:num_dof]
    sample_lo, sample_hi = cspace.T
    rng = np.random.default_rng(0)

    # Inverse kinematics.
    # def _run_inverse_kinematics(target_pos: Array, target_quat: Array) -> Array:
    #     sim = gym_env.sim
    #     q_init = rng.uniform(low=sample_lo, high=sample_hi)
    #     q_nom = np.zeros_like(q_init)
    #     body_pos = np.zeros_like(target_pos)
        
    #     # world_pos = target_pos
    #     # world_quat = target_quat
    #     world_pos = gym_env.get_ee_pose()
    #     world_quat = gym_env.get_ee_quat()
        
    #     body_id = sim.model.body_name2id('mocap')

    #     result = inverseKin(sim, q_init, q_nom, body_pos, world_pos, world_quat,
    #                         body_id, #upper=sample_hi, lower=sample_lo,
    #                         raise_on_fail=True)
    #     import ipdb; ipdb.set_trace()
    


    # def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    #     """Calculates the joint velocities to achieve a specified end effector delta.

    #     Args:
    #         jac_joints: The Jacobian of the end effector with respect to the joints. A
    #         numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
    #         and `nv` is the number of degrees of freedom.
    #         delta: The desired end-effector delta. A numpy array of shape `(3,)` or
    #         `(6,)` containing either position deltas, rotation deltas, or both.
    #         regularization_strength: (optional) Coefficient of the quadratic penalty
    #         on joint movements. Default is zero, i.e. no regularization.

    #     Returns:
    #         An `(nv,)` numpy array of joint velocities.

    #     Reference:
    #         Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
    #         transpose, pseudoinverse and damped least squares methods.
    #         https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    #     """
    #     hess_approx = jac_joints.T.dot(jac_joints)
    #     joint_delta = jac_joints.T.dot(delta)
    #     if regularization_strength > 0:
    #         # L2 regularization
    #         hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
    #         return np.linalg.solve(hess_approx, joint_delta)
    #     else:
    #         return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
    

    # # https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
    # def _run_inverse_kinematics(target_pos: Array, target_quat: Array,
    #                             max_steps: int = 100, rot_weight=1.0,
    #                                                     regularization_threshold=0.1,
    #                     regularization_strength=3e-2,
    #                     max_update_norm=0.01,
    #                     progress_thresh=20.0,
    #                             tol=1e-14) -> Array:
    #     # TODO remove?
    #     gym_env.reset()

    #     dtype = target_pos.dtype
    #     jac = np.empty((6, num_dof), dtype=dtype)
    #     err = np.empty(6, dtype=dtype)
    #     jac_pos, jac_rot = jac[:3], jac[3:]
    #     err_pos, err_rot = err[:3], err[3:]
    #     update_nv = np.zeros(num_dof, dtype=dtype)
    #     site_xquat = np.empty(4, dtype=dtype)
    #     neg_site_xquat = np.empty(4, dtype=dtype)
    #     err_rot_quat = np.empty(4, dtype=dtype)

    #     # Ensure that the Cartesian position of the site is up to date.
    #     # TODO: do anything?
    #     # mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

    #     # Convert site name to index.
    #     site_name = "end_effector"
    #     site_id = gym_env.sim.model.site_name2id(site_name)

    #     site_xpos = gym_env.get_site_xpos(site_name)
    #     site_xmat = gym_env.get_site_xmat(site_name)

    #     dof_indices = range(num_dof)  # TODO generalize?

    #     steps = 0
    #     success = False

    #     for steps in range(max_steps):

    #         err_norm = 0.0

    #         # Translational error.
    #         err_pos[:] = target_pos - site_xpos
    #         err_norm += np.linalg.norm(err_pos)

    #         # Rotational error.
    #         mujoco_py.functions.mju_mat2Quat(site_xquat, site_xmat)
    #         mujoco_py.functions.mju_negQuat(neg_site_xquat, site_xquat)
    #         mujoco_py.functions.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
    #         mujoco_py.functions.mju_quat2Vel(err_rot, err_rot_quat, 1)
    #         err_norm += np.linalg.norm(err_rot) * rot_weight

    #         if err_norm < tol:
    #             print(f'IK converged after {steps} steps: err_norm={err_norm}')
    #             success = True
    #             break
    #         else:
    #             # This is probably really slow, but I can't figure out a way around it.
    #             jacp = gym_env.data.get_site_jacp(site_name)
    #             jacr = gym_env.data.get_site_jacr(site_name)
    #             jac_pos[:] = jacp[:, :num_dof]
    #             jac_rot[:] = jacr[:, :num_dof]
    #             # mujoco_py.functions.mj_jacSite(gym_env.model.ptr, gym_env.data,
    #             #                                jac_pos, jac_rot, site_id)
    #             jac_joints = jac[:, dof_indices]

    #         reg_strength = (
    #             regularization_strength if err_norm > regularization_threshold
    #             else 0.0)
    #         update_joints = nullspace_method(
    #             jac_joints, err, regularization_strength=reg_strength)
            
    #         update_norm = np.linalg.norm(update_joints)

    #         # Check whether we are still making enough progress, and halt if not.
    #         progress_criterion = err_norm / update_norm
    #         if progress_criterion > progress_thresh:
    #             print(f'Step {steps}: err_norm / update_norm ({progress_criterion}) > '
    #                   f'tolerance ({progress_thresh}). Halting due to insufficient progress')
    #             break

    #         if update_norm > max_update_norm:
    #             update_joints *= max_update_norm / update_norm

    #         # Write the entries for the specified joints into the full `update_nv`
    #         # vector.
    #         update_nv[dof_indices] = update_joints

    #         # Again this is probably terrible...
    #         gym_env.set_robot_joint_position(update_nv)
    #         gym_env.render()

    #         print(f'Step {steps}: err_norm={err_norm} update_norm={update_norm}')

    #     import ipdb; ipdb.set_trace()
        


    # Set up RRT in 7 DOF space.
    action_magnitude = 0.1
    num_attempts = 10
    num_iters = 250
    smooth_amt = 0  # TODO
    sample_goal_eps = 0.0

    def _sample_fn(_: Array) -> Array:
        return rng.uniform(low=sample_lo, high=sample_hi)

    def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
        distance = np.linalg.norm(pt2 - pt1)
        num = int(distance / action_magnitude) + 1
        for i in range(1, num + 1):
            yield pt1 * (1 - i / num) + pt2 * i / num

    def _collision_fn(pt: Array) -> bool:
        return False  # TODO

    def _distance_fn(pt1: Array, pt2: Array) -> float:
        return np.sum(np.subtract(pt2, pt1)**2)
    
    def _sample_goal_fn() -> Array:
        # return _run_inverse_kinematics(target_pose, target_quat.elements)
        raise NotImplementedError
    
    def _check_goal_fn(pt: Array) -> bool:
        gym_env.set_robot_joint_position(pt)
        ee_pose = gym_env.get_ee_pose()
        ee_quat = Quaternion(gym_env.get_ee_quat())
        print(ee_pose, ee_quat)
        pose_goal_reached = np.sum(np.subtract(ee_pose, target_pose)**2) < target_pose_eps
        quat_goal_reached = Quaternion.absolute_distance(ee_quat, target_quat) < target_quat_eps
        goal_reached = pose_goal_reached and quat_goal_reached
        if goal_reached:
            if use_gui:
                gym_env.render()
            import ipdb; ipdb.set_trace()
        return goal_reached
    

    rrt = utils.RRT(_sample_fn, _extend_fn, _collision_fn, _distance_fn, rng,
                    num_attempts, num_iters, smooth_amt)
    
    result = rrt.query_to_goal_fn(start, _sample_goal_fn, _check_goal_fn, sample_goal_eps=sample_goal_eps)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    _main()
