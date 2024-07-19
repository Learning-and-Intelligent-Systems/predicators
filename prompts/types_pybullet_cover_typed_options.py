_block_type = Type("block", ["pose_y_norm", "bbox_left", "bbox_right", "bbox_upper", "bbox_lower"])
_target_type = Type("target", ["pose_y_norm", "bbox_left", "bbox_right", "bbox_upper", "bbox_lower"])
_robot_type = Type("robot", ["pose_y_norm", "pose_x", "pose_z", "bbox_left", 
                            "bbox_right", "bbox_upper", "bbox_lower"])
_table_type = Type("table", ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"])