class RawState:
    """
    A class representing the raw visual state of the world

    Attributes:
    -----------
    labeled_image : Image
        An observation of the state annotated with a unique label for each object.

    Methods:
    --------
    crop_to_objects(self, objects: Collection[Object],
                    left_margin: int = 5, lower_margin: int=10, 
                    right_margin: int=10, top_margin: int=5) -> Image:
        Crops the labeled image to only focus on the objects in the input.
    get(self, obj: Object, feature_name: str) -> Any:
        Looks up an object feature by name and returns the feature value.
    get_objects(self, object_type: Type) -> List[Object]:
        Returns objects of the given type.
    evaluate_simple_assertion(assertion: str, image: Image) -> bool:
        Evaluate a simple assertion about an image..
    """
    def crop_to_objects(self, objects: Collection[Object],
                        left_margin: int = 5,
                        lower_margin: int=10, 
                        right_margin: int=10, 
                        top_margin: int=5) -> Image:
        """
        Crop the labeled_image to only include the specified objects, with 
        optional margins around the objects.

        Parameters:
        -----------
        objects : Collection[Object]
            The objects to include in the cropped image.
        left_margin, lower_margin, right_margin, top_margin : int, optional
            The left, lower, right, and top margin to include in the cropped 
            image (default is 5, 10, 10 and 5).

        Returns:
        --------
        Image
            The cropped image.
        Examples:
        ---------
        >>> # An example for predicate On
        >>> def _On_NSP_holds(state: RawState, objects: Sequence[Object])\
        >>>     -> bool:
        >>>     '''
        >>>     Determine if the first block in objects is directly on top of 
        >>>     the second block 
        >>>     '''
        >>>     block1, block2 = objects
        >>>     ...
        >>>     # Crop the scene image to the smallest bounding box that include both objects.
        >>>     attention_image = state.crop_to_objects([block1, block2])
        >>>     return state.evaluate_simple_assertion(
        >>>         f"{block1_name} is directly on top of {block2_name} with no blocks in between.", attention_image)
        >>>
        >>> # An example for predicate OnTable
        >>> def _OnTable_NSP_holds(state: RawState, objects:Sequence[Object]) ->\
        >>>         bool:
        >>>     '''Determine if the block is directly resting on the table's 
        >>>     surface.
        >>>     '''
        >>>     apple, = objects
        >>>     apple_name = apple.id_name
        >>>     
        >>>     # Crop the scene image to the smallest bounding box that include both objects.
        >>>     # We know there is only one table in this environment.
        >>>     table = state.get_objects(_table_type)[0]
        >>>     table_name = table.id_name
        >>>     attention_image = state.crop_to_objects([apple, table])

        >>>     return state.evaluate_simple_assertion(
        >>>         f"{apple_name} is directly resting on {table_name}'s surface.",
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
        
        Example:
        ---------
        >>> # An example for predicate WristBent
        >>> _robot_type = Type("robot", ["x", "y", "tilt", "wrist", "fingers"])
        >>> def _WristBent_holds(state: State, objects: Sequence[Object]
        >>>                     ) -> bool:
        >>>     robot, = objects
        >>>     return state.get(robot, "wrist") >= 0.5
        >>> _WristBent = NSPredicate("WristBent", [_robot_type], _WristBent_holds)
        >>>
        >>> # An example for classifying Covers
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
        >>>        state.get(target, "bbox_right") > state.get(block, "bbox_right"):
        >>>         return False
        >>>     ...
        >>>     return state.evaluate_simple_assertion(...)
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

    def evaluate_simple_assertion(self, assertion: str, image: Image) -> bool:
        """
        A function that takes a simple assertion as a string and 
        an image as input, and returns a boolean indicating whether the 
        assertion holds true for the image according to the VLM.

        The assertion should be clear, unambiguous, and relatively simple, and 
        the image should have been cropped to only the relavant objects.

        Parameters:
        -----------
        assertion : str
            The assertion to be evaluated. This should be a clear, unambiguous, and 
            relatively simple statement about the image.

        image : Image
            The image for which the assertion is to be evaluated.

        Returns:
        --------
        bool
            True if the VLM determines that the assertion holds true for the image, 
            False otherwise.
        
        Examples:
        ---------
        >>> # An example for predicate Open
        >>> ...
        >>> return state.evaluate_simple_assertion(f"{door_name} is open", attention_image)

        >>> # An example for predicate CupOnTable
        >>> ...
        >>> return state.evaluate_simple_assertion(f"{cup_name} is resting on {shelf_name}", attention_image)

        >>> # An example for predicate CupFilled
        >>> ...
        >>> return state.evaluate_simple_assertion(f"{cup_name} is filled with liquid", attention_image)

        >>> # An example for predicate PluggedIn
        >>> ...
        >>> return state.evaluate_simple_assertion(f"{coffee_machine_name} is plugged into a socket", attention_image)
        """