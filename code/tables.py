#
# This file contains the structure for belief tables and the operations that can be executed on them
#
import numpy as np

import util


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
        if not (new_variables.keys() < self._variables.keys()):
            raise AttributeError("Variables to marginalize on must be a subset of variables of the table")

        new_table = np.zeros((2,)*len(new_variables))

        sum_variables = util.subtract_ordered_dict(self._variables, new_variables)

        # Create template for indexing, setting up which variables have to be extracted whole :
        # Marginalizing on AB over t_ABC means summing (:,:,0) + (:,:,1) (that is over C values)
        coord_template = []
        for el in self._variables:
            if el in new_variables:
                coord_template.append(slice(None))
            else:
                coord_template.append(0)

        # Iterate over all possible values of V\W
        for i in range(2**len(sum_variables)):
            bin_string = format(i, '0' + str(len(sum_variables)) + 'b')
            actual_coord = coord_template.copy()

            # Find the subtable by substituting the non ':' places in the index template with actual indexes
            index = 0     # Index for the binary string
            for ind, element in enumerate(actual_coord):
                if element != slice(None):
                    actual_coord[ind] = int(bin_string[index])
                    index += 1

            # Add each table
            new_table += self.get_prob(tuple(actual_coord))

        return BeliefTable(new_variables, new_table)

    def get_variable_index(self, variable):
        return list(self._variables.keys()).index(variable)

    def get_prob(self, coordinates):
        return self._table[coordinates]

    def get_prob_dict(self, variables):
        if variables.keys() != self._variables.keys():
            raise AttributeError("The variables in the dictionary were wrong")
        # Match with the right index
        coords = []
        for el in self._variables:
            coords.append(variables[el])

        return self.get_prob(tuple(coords))

    def get_variables(self):
        return list(self._variables)

    def set_probability_coord(self, coord, value):
        if len(coord) != len(self._variables):
            raise AttributeError("Wrong probability coordinate format")
        self._table[coord] = value

    def set_probability_dict(self, variables, value):
        if variables.keys() != self._variables.keys():
            raise AttributeError("The variables in the dictionary were wrong")
        # Match with the right index
        coords = []
        for el in self._variables:
            coords.append(variables[el])

        self.set_probability_coord(tuple(coords), value)


    def __str__(self):
        full_str = ''
        single_row = self._table.reshape(self._table.size)
        i = 0
        for x in np.nditer(single_row):
            bin_string = format(i, '0' + str(len(self._variables)) + 'b')
            j = 0
            for var in self._variables:
                full_str += var + ':' + bin_string[j] + ','
                j += 1
            full_str = full_str[:-1]
            full_str += '-> ' + str(x) + '\n'
            i += 1
        return full_str

