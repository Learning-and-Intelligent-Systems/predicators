class State:
    """A class representing the low-level state of the world.

    Attributes:
    -----------
    data : Dict[Object, Array]
        A dictionary mapping objects to their feature vectors. The feature vectors are numpy arrays.

    simulator_state : Optional[Any]
        Some environments may need to store additional simulator state. This field is provided for that purpose. It is optional and defaults to None.

    Methods:
    --------
    get(self, obj: Object, feature_name: str) -> Any:
        This method looks up an object feature by name. It returns the value of
        the feature.

    get_objects(self, object_type: Type) -> List[Object]:
        This method returns objects of the given type in the order of
        __iter__().
    """
    data: Dict[Object, Array]
    # Some environments will need to store additional simulator state, so
    # this field is provided.
    simulator_state: Optional[Any] = None

    def get(self, obj: Object, feature_name: str) -> Any:
        """Look up an object feature by name.

        Parameters:
        -----------
        obj : Object
            The object whose feature value is to be retrieved.
        feature_name : str
            The name of the feature to be retrieved.

        Returns:
        --------
        Any
            The value of the specified feature for the given object.

        Raises:
        -------
        ValueError
            If the specified feature name is not found in the object's type feature names.
        
        Examples:
        ---------
        >>> # An example for predicate Covers
        >>> _block_type = Type("block", ["is_block", "is_target", "width", 
                "pose", "grasp"])
        >>> _target_type = Type("target", ["is_block", "is_target", "width", 
                "pose"])
        >>> block1 = Object("block1", _block_type)
        >>> target1 = Object("target1", _target_type)
        >>> state = State({
                block1: np.array([1.0, 0.0, 0.1, 0.2, -1.0]), 
                target1: np.array([0.0, 1.0, 0.05, 0.4])})
        >>> def _Covers_holds(state: State, objects: Sequence[Object]) -> 
                    bool:
        >>>     block, target = objects
        >>>     block_pose = state.get(block, "pose")
        >>>     block_width = state.get(block, "width")
        >>>     target_pose = state.get(target, "pose")
        >>>     target_width = state.get(target, "width")
        >>>     return (block_pose-block_width/2 <= \
                        target_pose-target_width/2) and \
                        (block_pose+block_width/2 >= \
                        target_pose+target_width/2) and \
                        state.get(block, "grasp") == -1
        >>> _Covers = Predicate("Covers", [_block_type, _target_type],
                        _Covers_holds)

        >>> # Another example for predicate On
        >>> _block_type = Type("block", ["pose_x", "pose_y", "pose_z", 
                            "held", "color_r", "color_g", "color_b"])
        >>> block1 = Object("block1", _block_type)
        >>> block2 = Object("block2", _block_type)
        >>> state = State({
                block1: np.array([1.0, 3.0, 0.2, 0.0, 1.0, 0.0, 0.0]),
                block2: np.array([2.0, 3.0, 0.3, 0.0, 0.0, 1.0, 0.0])})
        >>> on_tol = 0.01
        >>> def _On_holds(self, state: State, objects: Sequence[Object]) ->\ 
                bool:
        >>>     block1, block2 = objects
        >>>     if state.get(block1, "held") >= self.held_tol or \
        >>>        state.get(block2, "held") >= self.held_tol:
        >>>         return False
        >>>     x1 = state.get(block1, "pose_x")
        >>>     y1 = state.get(block1, "pose_y")
        >>>     z1 = state.get(block1, "pose_z")
        >>>     x2 = state.get(block2, "pose_x")
        >>>     y2 = state.get(block2, "pose_y")
        >>>     z2 = state.get(block2, "pose_z")
        >>>     return np.allclose([x1, y1, z1], 
                        [x2, y2, z2 + self._block_size],
                        atol=on_tol)
        >>> _On = Predicate("On", [_block_type, _block_type],
                            _On_holds)
        """

    def get_objects(self, object_type: Type) -> List[Object]:
        """Return objects of the given type in the order of __iter__().

        Parameters:
        -----------
        object_type : Type
            The type of the objects to be retrieved.

        Returns:
        --------
        List[Object]
            A list of objects of the specified type, in the order they are 
            iterated over in the state.

        Examples:
        ---------
        >>> _robot_type = Type("robot",
                                ["x", "y", "z", "tilt", "wrist", "fingers"])
        >>> _cup_type = Type("cup",
            ["x", "y", "capacity_liquid", "target_liquid", "current_liquid"])
        >>> robot = Object("robby", _robot_type)
        >>> cup1 = Object("cup1", _cup_type)
        >>> cup2 = Object("cup2", _cup_type)
        >>> state = State({
                        robot: np.array([5.0, 5.0, 10.0, 0.0, 0.0, 0.4]),
                        cup1: np.array([3.0, 2.0, 1.0, 0.75, 0.0]),
                        cup2: np.array([5.0, 6.0, 1.5, 1.125, 0.0])})
        >>> def _NotAboveCup_holds(state: State,
        >>>                        objects: Sequence[Object]) -> bool:
        >>>     robot, jug = objects
        >>>     for cup in state.get_objects(_cup_type):
        >>>         if _robot_jug_above_cup(state, cup):
        >>>             return False
        >>>     return True
        >>> _NotAboveCup = Predicate("NotAboveCup", [_robot_type, _jug_type],
                                    _NotAboveCup_holds)
        """