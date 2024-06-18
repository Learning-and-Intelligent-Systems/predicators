class RawState:
    """
    A class representing the raw visual state of the world

    Attributes:
    -----------
    labeled_image : Image
        An observation of the state of the world annotated with a unique label 
        for each object.

    Methods:
    --------
    crop_to_objects(self, objects: Collection[Object],
                    left_margin: int = 5, lower_margin: int=10, 
                    right_margin: int=10, top_margin: int=5) -> Image:
        Crops the labeled image to only focus on the objects in the input.
    get(self, obj: Object, feature_name: str) -> Any:
        This method looks up an object feature by name. It returns the value of 
        the feature.
    get_objects(self, object_type: Type) -> List[Object]:
        This method returns objects of the given type in the order of 
        __iter__().
    """
    def crop_to_objects(self, objects: Collection[Object],
                        left_margin: int = 5,
                        lower_margin: int=10, 
                        right_margin: int=10, 
                        top_margin: int=5) -> Image:
        """
        Crop the labeled image observation of the state to only include the 
        specified objects.

        The cropping is done by utilizing the masks of the objects, with optional 
        margins around the objects.

        Parameters:
        -----------
        objects : Collection[Object]
            The objects to include in the cropped image.
        left_margin : int, optional
            The left margin to include in the cropped image (default is 5).
        lower_margin : int, optional
            The lower margin to include in the cropped image (default is 10).
        right_margin : int, optional
            The right margin to include in the cropped image (default is 10).
        top_margin : int, optional
            The top margin to include in the cropped image (default is 5).

        Returns:
        --------
        Image
            The cropped image.
        Example 1:
        ----------
        >>> # An example for predicate On
        >>> def _On_NSP_holds(state: RawState, objects: Sequence[Object])\
        >>>     -> bool:
        >>>     '''
        >>>     Determine if the first block in objects is directly on top of the second 
        >>>     block in the scene image, by using simple heuristics and image processing 
        >>>     techniques.
        >>>     '''
        >>>     block1, block2 = objects
        >>>     ...
        >>>     # Crop the scene image to the smallest bounding box that include both objects.
        >>>     attention_image = state.crop_to_objects([block1, block2])
        >>>
        >>>     return evaluate_simple_assertion(
        >>>         f"{block1_name} is directly on top of {block2_name} with no blocks in between.", attention_image)
        
        Example 2: 
        ----------
        >>> # An example for predicate OnTable
        >>> def _OnTable_NSP_holds(state: RawState, objects:Sequence[Object]) ->\
        >>>         bool:
        >>>     '''Determine if the block in objects is directly resting on the table's 
        >>>     surface in the scene image.
        >>>     '''
        >>>     block, = objects
        >>>     block_name = block.id_name
        >>>     
        >>>     # Crop the scene image to the smallest bounding box that include both objects.
        >>>     # We know there is only one table in this environment.
        >>>     table = state.get_objects(_table_type)[0]
        >>>     table_name = table.id_name
        >>>     attention_image = state.crop_to_objects([block, table])

        >>>     return evaluate_simple_assertion(
        >>>         f"{block_name} is directly resting on {table_name}'s surface.",
        >>>         attention_image)
        """

    def get(self, obj: Object, feature_name: str) -> Any:
        """
        Look up an object feature by name.

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
        
        Example 1:
        ---------
        >>> _robot_type = Type("robot", ["x", "y", "tilt", "wrist", "fingers"])
        >>> def _WristBent_holds(state: State, objects: Sequence[Object]
        >>>                     ) -> bool:
        >>>     robot, = objects
        >>>     return state.get(robot, "wrist") >= 0.5
        >>> _WristBent = NSPredicate("WristBent", [_robot_type], _WristBent_holds)

        Example 2:
        --------
        >>> An example for classifying Covers
        >>> def _Covers_NSP_holds(state: State, objects: Sequence[Object]
        >>>                         ) -> bool:
        >>>     '''
        >>>     Determine if the block is covering (directly on top of) the target 
        >>>     region.
        >>>     '''
        >>>     block, target = objects
        >>>
        >>>     # Necessary but not sufficient condition for covering: no part of the 
        >>>     # target region is outside the block.
        >>>     if state.get(target, "bbox_left") < state.get(block, "bbox_left") or\
                   state.get(target, "bbox_right") > state.get(block, "bbox_right"):
        >>>         return False
        >>>     ...
        >>>     return evaluate_simple_assertion(...)
        """

    def get_objects(self, object_type: Type) -> List[Object]:
        """
        Return objects of the given type in the state

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
        >>> def _robot_hand_above_cup(state: State, cup: Object) -> bool:
        >>>     ...
        >>>
        >>> def _HandNotAboveCup_holds(state: State,
        >>>                            objects: Sequence[Object]) -> bool:
        >>>     for cup in state.get_objects(_cup_type):
        >>>         if _robot_hand_above_cup(state, cup):
        >>>             return False
        >>>     return True
        >>> _HandNotAboveCup = NSPredicate("HandNotAboveCup", [], 
        >>>                              _HandNotAboveCup_holds)
        """
