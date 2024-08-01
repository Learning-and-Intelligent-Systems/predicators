_object_type = Type("object", [])
_item_type = Type("item", [], parent=_object_type)
_station_type = Type("station", [], parent=_object_type)

_robot_type = Type("robot", ["row", "col", "z", "fingers", "dir"],
                   parent=_object_type)

_patty_type = Type("patty", ["row", "col", "z"], parent=_item_type)
_tomato_type = Type("tomato", ["row", "col", "z"], parent=_item_type)
_cheese_type = Type("cheese", ["row", "col", "z"], parent=_item_type)
_bottom_bun_type = Type("bottom_bun", ["row", "col", "z"], parent=_item_type)
_top_bun_type = Type("top_bun", ["row", "col", "z"], parent=_item_type)

_grill_type = Type("grill", ["row", "col", "z"], parent=_station_type)
_cutting_board_type = Type("cutting_board", ["row", "col", "z"],
                           parent=_station_type)
