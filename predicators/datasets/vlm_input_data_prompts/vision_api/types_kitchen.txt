object_type = Type("object", ["x", "y", "z"])
gripper_type = Type("gripper", ["x", "y", "z", "qw", "qx", "qy", "qz"],
                    parent=object_type)
on_off_type = Type("on_off", ["x", "y", "z", "angle"], parent=object_type)
hinge_door_type = Type("hinge_door", ["x", "y", "z", "angle"],
                       parent=on_off_type)
knob_type = Type("knob", ["x", "y", "z", "angle"], parent=on_off_type)
switch_type = Type("switch", ["x", "y", "z", "angle"], parent=on_off_type)
surface_type = Type("surface", ["x", "y", "z"], parent=object_type)
kettle_type = Type("kettle", ["x", "y", "z"], parent=object_type)
