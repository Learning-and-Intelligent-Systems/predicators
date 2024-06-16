class NSPredicate:
    """
    A class representing a predicate, a classifier that characterizes properties 
    of states in the context of AI task planning.
    A predicate is a function that takes a state and a sequence of objects as 
    input, and returns a boolean value indicating whether a certain  property 
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

def evaluate_simple_assertion(assertion: str, image: Image) -> bool:
    """
    Evaluate a simple assertion about an image by querying a vision language 
    model (VLM).

    This function is a helper that can be used in writing _classifier functions 
    for NSPredicates. It takes a simple assertion as a string and an image as 
    input, and returns a boolean value indicating whether the assertion holds 
    true for the image according to the VLM.

    Note that VLM has limited visual understanding so the assertion should be 
    clear, unambiguous, and relatively simple, and the image should have been 
    cropped to only the relavant objects.
    Moreover, don't write heuristics or rules that are not always true.

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
    """