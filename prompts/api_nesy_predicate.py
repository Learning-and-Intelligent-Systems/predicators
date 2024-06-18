class NSPredicate:
    """
    A class representing a predicate, a classifier that characterizes properties 
    of states in the context of AI task planning.
    A predicate is a function that takes a state and a sequence of objects as 
    input, and returns a boolean value indicating whether a certain property 
    holds for those objects in that state.

    Parameters:
    -----------
    name : str
        The name of the predicate.

    types : Sequence[Type]
        The types of the objects that the predicate applies to. This sequence 
        length should match the number of objects passed to the classifier. Each
        type corresponds one-to-one with an object in the sequence. 

    _classifier : Callable[[State, Sequence[Object]], bool]
        The classifier function for the predicate. It takes a state and a
        sequence of objects as input, and returns a boolean value. The sequence
        of objects should correspond one-to-one with the 'types' attribute. The 
        classifier returns True if the predicate holds for those objects in that 
        state, and False otherwise.
    """    