class RawState:
    """
    A class representing the raw visual state of the world

    Attributes:
    -----------
    labeled_image : PIL.Image.Image
        An observation of the state of the world annotated with an unique label 
        for each object.
    obj_mask_dict : Dict[str, Mask]
        A dictionary mapping object names to their corresponding segmentation 
        mask.

    Methods:
    --------
    get_obj_bbox(self, object: Object) -> BoundingBox:
        Returns the bounding box of the object in the state image.
    crop_to_objects(self, objects: Collection[Object],
                    left_margin: int = 5, lower_margin: int=10, 
                    right_margin: int=10, top_margin: int=5) -> Image:
        Crops the labeled image to only focus on the objects in the input.
    """
    def get_obj_bbox(self, object: Object) -> BoundingBox:
        """
        Get the bounding box of the specified object in the labeled image.
        The bounding box is defined by the column and row indices of the 
        object's boundaries in the state image. The (0, 0) index starts from the
        bottom left corner of the original image.

        Parameters:
        -----------
        object : Object
            The object for which to get the bounding box.

        Returns:
        --------
        BoundingBox
            The bounding box of the specified object, with attribute `left`
            `lower`, `right`, and `top` representing the column and row indices
            of the object's boundaries in the labeled image.

        Example:
        --------
        >>> # An example for predicate On
        >>> _block_type = Type("block", [])
        >>> def _On_NSP_holds(self, state: RawState, objects: Sequence[Object])\
        >>>     -> bool:
        >>>     '''
        >>>     Determine if the first block in objects is directly on top of the second 
        >>>     block in the scene image, by using simple heuristics and image processing 
        >>>     techniques.
        >>>
        >>>     Parameters:
        >>>     -----------
        >>>     state : RawState
        >>>     The current state of the world, represented as an image.
        >>>     objects : Sequence[Object]
        >>>     A sequence of two blocks whose relationship is to be determined. The 
        >>>     first block is the one that is potentially on top.
        >>>
        >>>     Returns:
        >>>     --------
        >>>     bool
        >>>     True if the first block is directly on top of the second block with 
        >>>     no blocks in between, False otherwise.
        >>>     '''
        >>>
        >>>     block1, block2 = objects
        >>>     block1_name, block2_name = block1.id_name, block2.id_name
        >>>
        >>>     # Heuristics: we know a block can't be on top of itself.
        >>>     if block1_name == block2_name:
        >>>         return False
        >>>
        >>>     # Using simple heuristics to check if they are far away
        >>>     block1_bbox = state.get_obj_bbox(block1) 
        >>>     block2_bbox = state.get_obj_bbox(block2)
        >>>     if (block1_bbox.lower < block2_bbox.lower) or \
        >>>        (block1_bbox.left > block2_bbox.right) or \
        >>>        (block1_bbox.right < block2_bbox.left) or \
        >>>        (block1_bbox.upper < block2_bbox.upper):
        >>>         return False
        >>>
        >>>     # Crop the scene image to the smallest bounding box that include both
        >>>     # objects.
        >>>     attention_image = state.crop_to_objects([block1, block2])
        >>>
        >>>     return evaluate_simple_assertion(
        >>>      f"{block1_name} is directly on top of {block2_name} with no blocks"+
        >>>      " in between.", attention_image)
        >>> _On_NSP = NSPredicate("On", [_block_type, _block_type],
        >>>                         _On_NSP_holds)
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
        
        Example:
        --------
        >>> # An example for predicate OnTable
        >>> _block_type = Type("block", [])
        >>> _table_type = Type("table", [])
        >>> def _OnTable_NSP_holds(state: RawState, objects:Sequence[Object]) ->\
        >>>         bool:
        >>>     '''Determine if the block in objects is directly resting on the table's 
        >>>     surface in the scene image.

        >>>     Parameters:
        >>>     -----------
        >>>     state : RawState
        >>>         The current state of the world, represented as an image.
        >>>     objects : Sequence[Object]
        >>>         A sequence containing a single block whose relationship with the 
        >>>         table is to be determined.

        >>>     Returns:
        >>>     --------
        >>>     bool
        >>>         True if the block is directly resting on the table's surface, False 
        >>>         otherwise.
        >>>     '''
        >>>     block, = objects
        >>>     block_name = block.id_name
        >>>     
        >>>     # Crop the scene image to the smallest bounding box that include both
        >>>     # objects.
        >>>     # We know there is only one table in this environment.
        >>>     table = state.get_objects(_table_type)[0]
        >>>     table_name = table.id_name
        >>>     attention_image = state.crop_to_objects([block, table])

        >>>     return evaluate_simple_assertion(
        >>>         f"{block_name} is directly resting on {table_name}'s surface.",
        >>>         attention_image)
        >>> _OnTable_NSP = NSPredicate("OnTable", [_block_type], 
        >>>                 _OnTable_NSP_holds)
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
        
        Examples:
        ---------
        >>> _robot_type = Type("robot", ["x", "y", "tilt", "wrist", "fingers"])
        >>> def _HandOpen_holds(self, state: State, objects: Sequence[Object]
        >>>                     ) -> bool:
        >>>     robot, = objects
        >>>     return state.get(robot, "fingers") == 1.0
        >>> _HandOpen = NSPredicate("HandOpen", [_robot_type], _HandOpen_holds)
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
        >>> def _HandNotAboveCup_holds(state: State,
        >>>                        objects: Sequence[Object]) -> bool:
        >>>     for cup in state.get_objects(_cup_type):
        >>>         if _robot_hand_above_cup(state, cup):
        >>>             return False
        >>>     return True
        >>> _HandNotAboveCup = NSPredicate("HandNotAboveCup", [], 
        >>>                              _HandNotAboveCup_holds)
        """