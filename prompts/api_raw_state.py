class RawState:
    """
    A class representing the raw visual state of the world

    Attributes:
    -----------
    state_image : PIL.Image.Image
    obj_mask_dict : Dict[str, Mask]
        A dictionary mapping object names to their corresponding Mask.
    
    Examples:
    ---------
    >>> # An example for predicate Covers
    >>> _block_type = Type("block", ["is_block", "is_target", "width", 
            "pose", "grasp"])
    >>> _target_type = Type("target", ["is_block", "is_target", "width", 
            "pose"])
    >>> block1 = Object("block1", _block_type)
    >>> target1 = Object("target1", _target_type)
    >>> state: PyBulletRenderedState = render_state({
            block1: np.array([1.0, 0.0, 0.1, 0.2, -1.0]), 
            target1: np.array([0.0, 1.0, 0.05, 0.4])})
    >>> ...
    """
    state_image: PIL.Image.Image
    obj_mask_dict: Dict[str, Mask] = field(default_factory=dict)

    def get_scene_image(self) -> PIL.Image.Image:
        """
        Get the full scene image.

        Returns:
        --------
        Image
            The full scene image.
        """

#     def get_object_image(self, obj: Object) -> ImageWithBox:
#         """
#         Return the ImageWithBox object correspond to that object

#         Parameters:
#         -----------
#         obj : Object
#             The object whose image is to be retrieved.
#         """

    def get_objects(self, object_type: Type) -> List[Object]:
        """
        Return objects of the given type in the order of __iter__().

        Parameters:
        -----------
        object_type : Type
            The type of the objects to be retrieved.

        Returns:
        --------
        List[Object]
            A list of objects of the specified type, in the order they are 
            iterated over in the state.
        """