class RawState:
    """A class representing the raw visual state of the world.
    Methods:
    --------
    get_objects(self, object_type: Type) -> List[Object]:
        Returns objects of the given type.
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
        >>> def _AppleInBowl_NSP_holds(state: RawState, objects: Sequence[Object]) -> bool:
        >>>     apple, bowl = objects
        >>>     ...
        >>>
        >>> def _BowlEmpty_NSP_holds(state: RawState,
        >>>                            objects: Sequence[Object]) -> bool:
        >>>     bowl, = objects
        >>>     for apple in state.get_objects(_apple_type):
        >>>         if _AppleInBowl_NSP_holds(state, [bowl]):
        >>>             return False
        >>>     return True
        >>> BowlEmpty = NSPredicate("BowlEmpty", [_bowl_type], 
        >>>                              _BowlEmpty_NSP_holds)
        """