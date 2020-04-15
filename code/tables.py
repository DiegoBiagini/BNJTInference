#
# This file contains the structure for belief tables and the operation that can be executed on them
#


class BeliefTable(object):
    _variables = {}
    _table = None

    def __init__(self, variables, table):
        """
        Args:
            variables: A dictionary that represents the variables contained in the table
            table: A numpy multidimensional array, its size should be 2^^(number of variables)
        """
        self._variables = variables
        self._table = table

    def multiply(self, t2):
        """
        Args:
            t2: The second term of the multiplication
        """
        pass

    def divide(self, t2):
        """
        Args:
            t2: The divider
        """
        pass

    def marginalize(self, new_variables):
        """
        Args:
            new_variables: A dictionary containing the variables to marginalize on
        """
        pass
