#
# General utility functions
#


def subtract_ordered_dict(d1, d2):
    """
    Executes set subtraction between two dictionaries used as ordered sets

    :type d1: dict
    :type d2: dict
    :return: the result of set subtraction
    :rtype: dict
    """

    return dict.fromkeys([el for el in d1.keys() if el not in d2.keys()])
