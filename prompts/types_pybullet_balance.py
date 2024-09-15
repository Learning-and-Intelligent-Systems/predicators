bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
_block_type = Type("block", [
    "pose_x", "pose_y", "pose_z", "held", "color_r", "color_g",
    "color_b"] + bbox_features)
_robot_type = Type(
    "robot", ["pose_x", "pose_y", "pose_z", "fingers"] + bbox_features)
_table_type = Type("table", bbox_features)
_machine_type = Type("machine", ["is_on"] + bbox_features)