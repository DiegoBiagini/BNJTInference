#
# General utility functions
#


def subtract_ordered_dict(d1, d2):
    """
    Executes set subtraction between two dictionaries used as ordered sets and returns the result
    Args:
        d1:first subtraction term
        d2:second subtraction term
    """
    return dict.fromkeys([el for el in d1.keys() if el not in d2.keys()])
