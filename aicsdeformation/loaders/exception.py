
class MinBeadMatchNotMetException(Exception):
    """
    This exception class is intended for when insufficient time-points are provided
    for temporal deformation calculations.
    """
    def __init__(self, match_count: int, req_match_count: int):
        super().__init__()
        self.match_count = match_count
        self.required_match_count = req_match_count

    def __str__(self):
        return f"The number of matched points was insufficent to apply findHomography, " \
               f"({self.match_count} > {self.required_match_count} VIOLATED)"


class InsufficientTimePointsException(Exception):
    """
    Exception class for problems like the class being passed a single time-point
    """
    pass
