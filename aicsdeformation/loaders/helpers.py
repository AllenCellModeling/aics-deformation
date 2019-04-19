
def raise_a_if_b(a: Exception, b: bool) -> None:
    """
    This is a helper function to allow me to raise an exception without obscuring the flow
    of a the functions intent with an if block
    :param a: The exception to raise if the condition is true
    :param b: The condition (evaluating to True/False) to test True => Raise exception
    :return: None
    """
    if b:
        raise a
