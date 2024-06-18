class Predicate:
    """
    A class representing a predicate (a lifted classifier over states) in the 
    context of AI task planning.
    The function takes a state and a sequence of objects as input, 
    and returns a boolean value indicating whether the property holds for those 
    objects in that state.

    Attributes:
    -----------
    name : str
        The name of the predicate.

    types : Sequence[Type]
        The types of the objects that the predicate applies to. This sequence 
        should have the same length as the sequence of objects passed to the 
        classifier.

    _classifier : Callable[[State, Sequence[Object]], bool]
        The classifier function for the predicate. This function takes a state 
        and a sequence of objects as input, and returns a boolean value. The 
        objects in the sequence should correspond one-to-one with the types in 
        the 'types' attribute. The classifier should return True if the 
        predicate holds for those objects in that state, and False otherwise.
    """    
    name: str
    types: Sequence[Type]
    _classifier: Callable[[State, Sequence[Object]], bool]