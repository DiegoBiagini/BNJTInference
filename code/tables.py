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

        # Merge all variables
        new_variables = {**self._variables, **t2._variables}

        # Create empty table
        new_shape = (2,) * len(new_variables)
        new_table = np.empty(new_shape)

        for index, value in np.ndenumerate(new_table):
            # Find the corresponding indexes in the 2 multiplying terms
            # For example: t_{A,B}*t_{A,C}=t_{A,B,C}
            # the element in position (a,b,c) in the result is the product of the elements in positions
            # (a,b) and in position (a,c)

            first_index = tuple([x for i, x in enumerate(index) if list(new_variables.keys())[i] in self._variables.keys()])
            second_index = tuple([x for i, x in enumerate(index) if list(new_variables.keys())[i] in t2._variables.keys()])

            new_table[index] = self._table[first_index] * t2._table[second_index]

        return BeliefTable(new_variables, new_table)

    def divide(self, t2):
        """
        Args:
            t2: The divider
        """
        # Merge all variables
        new_variables = {**self._variables, **t2._variables}

        # Create empty table
        new_shape = (2,) * len(new_variables)
        new_table = np.empty(new_shape)

        for index, value in np.ndenumerate(new_table):
            # Find the corresponding indexes in the 2 multiplying terms
            first_index = tuple([x for i, x in enumerate(index) if list(new_variables.keys())[i] in self._variables.keys()])
            second_index = tuple([x for i, x in enumerate(index) if list(new_variables.keys())[i] in t2._variables.keys()])

            new_table[index] = self._table[first_index] / t2._table[second_index]

        return BeliefTable(new_variables, new_table)

    def marginalize(self, new_variables):
        """
        Args:
            new_variables: A dictionary containing the variables to marginalize on
        """
        pass

    def get_prob(self, coordinates):
        return self._table[coordinates]

    def __str__(self):
        return str(self._variables) + '\n' + str(self._table)

