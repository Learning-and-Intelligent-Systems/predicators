class RawState:
    """A class representing the raw visual state of the world.

    Attributes:
    -----------
    labeled_image : Image
        An observation of the state annotated with a unique label for each object.

    Methods:
    --------
    crop_to_objects(self, objects: Sequence[Object],
                    left_margin: int = 30, lower_margin: int = 30,
                    right_margin: int = 30, top_margin: int = 30) -> Image:
        Crops the labeled image to only focus on the objects in the input.
    get(self, obj: Object, feature_name: str) -> Any:
        Looks up an object feature by name and returns the feature value.
    get_objects(self, object_type: Type) -> List[Object]:
        Returns objects of the given type.
    evaluate_simple_assertion(assertion: str, image: Image) -> bool:
        Evaluate a simple assertion about an image..
    """
    def crop_to_objects(self, objects: Sequence[Object],
                        left_margin: int = 30, lower_margin: int = 30, 
                        right_margin: int = 30, top_margin: int = 30) -> Image:
        """Crop the labeled_image to only include the specified objects, with
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
        >>> # An example for predicate SwitchOn
        >>> def _SwitchOn_NSP_holds(state: RawState, objects: Sequence[Object])\
        >>>     -> bool:
        >>>     '''
        >>>     Determine if the light switch is in the on state.
        >>>     '''
        >>>     switch, = objects
        >>>     switch_name = switch.id_name
        >>>     ...
        >>>     # Crop the scene image to focus.
        >>>     attention_image = state.crop_to_objects([switch], left_margin=20, 
        >>>                         right_margin=20)
        >>>     return state.evaluate_simple_assertion(
        >>>         f"{switch_name} is in the on state.", attention_image)
        >>>
        >>> # An example for predicate OnTable
        >>> def _OnTable_NSP_holds(state: RawState, objects:Sequence[Object]) ->\
        >>>         bool:
        >>>     '''Determine if the object is directly resting on the table's 
        >>>     surface.
        >>>     '''
        >>>     phone, = objects
        >>>     phone_name = phone.id_name
        >>>     
        >>>     # Crop the scene image to the smallest bounding box that include both objects.
        >>>     # We know there is only one table in this environment.
        >>>     table = state.get_objects(_table_type)[0]
        >>>     table_name = table.id_name
        >>>     attention_image = state.crop_to_objects([phone, table])

        >>>     return state.evaluate_simple_assertion(
        >>>         f"{phone_name} is directly resting on {table_name}'s surface.",
        >>>         attention_image)
        """

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
        
        Example:
        ---------
        >>> # An example for predicate GripperOpen
        >>> _robot_type = Type("robot", ["x", "y", "tilt", "wrist", "gripper"])
        >>> def _GripperOpen_holds(state: State, objects: Sequence[Object]
        >>>                     ) -> bool:
        >>>     robot, = objects
        >>>     return state.get(robot, "gripper") > 0.5
        >>> _GripperOpen = NSPredicate("GripperOpen", [_robot_type], 
        >>>                             _GripperOpen_holds)
        >>>
        >>> # An example for predicate Covers
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
        >>>     block_id, target_id = block.id_name, target.id_name
        >>>     attention_image = state.crop_to_objects([block, target])
        >>>     return state.evaluate_simple_assertion(
        >>>         f"{block_id} is on top of and covering {target_id}.", 
        >>>         attention_image)
        >>> _Covers = NSPredicate("Covers", [_block_type, _target_type],
        >>>                       _Covers_holds)
        """

    def get_objects(self, object_type: Type) -> List[Object]:
        """Return all objects of the given type in the state.

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
        >>> def _AppleInBowl_holds(state: State, objects: Sequence[Object]) -> bool:
        >>>     apple, bowl = objects
        >>>     ...
        >>>
        >>> def _BowlEmpty_holds(state: State,
        >>>                            objects: Sequence[Object]) -> bool:
        >>>     bowl, = objects
        >>>     for apple in state.get_objects(_apple_type):
        >>>         if _AppleInBowl_holds(state, [bowl]):
        >>>             return False
        >>>     return True
        >>> _BowlEmpty = NSPredicate("BowlEmpty", [], 
        >>>                              _BowlEmpty_holds)
        """

    def evaluate_simple_assertion(self, assertion: str, image: Image) -> bool:
        """A function that takes a simple assertion as a string and an image as
        input, and returns a boolean indicating whether the assertion holds
        true for the image according to the VLM.

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
        >>> return state.evaluate_simple_assertion(f"{door_name} is open", 
        >>>                                         attention_image)

        >>> # An example for predicate CupOnShelf
        >>> ...
        >>> return state.evaluate_simple_assertion(
        >>>             f"{cup_name} is resting on {shelf_name}", 
        >>>             attention_image)

        >>> # An example for predicate CupHasTeaBag
        >>> ...
        >>> return state.evaluate_simple_assertion(
        >>>             f"{cup_name} has a tea bag in it", attention_image)

        >>> # An example for predicate PluggedIn
        >>> ...
        >>> return state.evaluate_simple_assertion(
        >>>             f"{coffee_machine_name} is plugged into a socket", 
        >>>             attention_image)

        >>> # An example for predicate Cooked
        >>> ...
        >>> return state.evaluate_simple_assertion(
        >>>             f"{patty_name} is not raw and cooked", 
        >>>             attention_image)
        """