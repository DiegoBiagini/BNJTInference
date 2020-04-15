#
# This file contains the structure for belief tables and the operations that can be executed on them
#
import numpy as np

class BeliefTable(object):
    _variables = None
    _table = None

    def __init__(self, variables, table):
        """
        Args:
            variables: A dictionary that represents the variables contained in the table
            table: A numpy multidimensional array, its size should be 2^^(number of variables)
        """
        self._variables = variables
        self._table = table

        # Check table size
        if 2 ** len(self._variables) != self._table.size:
            raise AttributeError("Wrong array size")

    def multiply(self, t2):
        """
        Args:
            t2: The second term of the multiplication
        """
        if self._variables.keys() == t2._variables.keys():
            # Same variables, term by term multiplication
            new_var = dict(self._variables)
            new_table = np.empty(self._table.shape)

            for index, value in np.ndenumerate(new_table):
                new_table[index] = self._table[index] * t2._table[index]

            return BeliefTable(new_var, new_table)


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

    def __str__(self):
        return str(self._variables) + '\n' + str(self._table)

