from predicators.src.pybullet_helpers.utils import get_joint_info


def get_ordered_ancestors(robot, link, physics_client_id):
    # return prune_fixed_joints(robot, get_link_ancestors(robot, link)[1:] + [link])
    return get_link_ancestors(robot, link, physics_client_id)[1:] + [link]


def get_link_ancestors(body, link, physics_client_id):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(body, link, physics_client_id)
    if parent is None:
        return []
    return get_link_ancestors(body, parent, physics_client_id) + [parent]


def get_link_parent(body, link, physics_client_id):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, physics_client_id).parentIndex


BASE_LINK = -1
