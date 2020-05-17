#
# This file contains the structure for belief tables and the operations that can be executed on them
#
import numpy as np

import util


class BeliefTable(object):
    """
    The representation of a Belief Table, that is a set of variables(for which a value can be assigned) and a table that
     contains an entry for each combination of the various variables values.
     These entries represent different relationships depending on the usage of the BeliefTable.
    """

    _variables = None
    """
    A dictionary of variables,used as an ordered set. Each variable is represented by a key(of type string)
    in the dictionary with None as its value.
    """
    _table = None
    """
    A numpy table that contains all the entries of the BeliefTable. 
    For simplicity each variable can only take True or False as its values, so the size of the table is always 2^n, 
    where n is the number of variables
    """

    def __init__(self, variables, table=None):
        """
        Initializes the BeliefTable with the variables and optionally their table, if no table is given a 2^n table of
        zeros will be used

        :type table: np.ndarray
        :type variables: list[string] or dict[string,None]
        """
        self._variables = dict.fromkeys(variables)
        if table is not None:
            self._table = table

            # Check table size
            if 2 ** len(self._variables) != self._table.size:
                raise AttributeError("Wrong array size")
        else:
            self._table = np.zeros((2,) * len(variables))

    def multiply_table(self, t2):
        """
        Performs BeliefTable multiplication, the two steps are
        -create a new table that contains all variables of both
        tables
        -fill each entry of the table like this for example:  t_{A=0,B=1,C=0} = t_{A=0,B=1}*t_{A=0,C=1}

        :param t2: The second multiplication term
        :type t2: BeliefTable
        :return: the result of the multiplication
        :rtype: BeliefTable
        """

        # Merge all variables
        new_variables = {**self._variables, **t2._variables}

        # Create empty table
        new_shape = (2,) * len(new_variables)
        new_table = np.empty(new_shape)

        for index, value in np.ndenumerate(new_table):
            # Find the corresponding indexes in the 2 multiplying terms
            # For example: t_{A,B}*t_{A,C}=t_{A,B,C}
            # the element in position (0,1,0) in the result is the product of the elements in positions
            # (0,1) and in position (0,0)
            dict_1_proxy = dict.fromkeys(self._variables)
            dict_2_proxy = dict.fromkeys(t2._variables)

            i = 0
            for variable in list(new_variables.keys()):
                if variable in dict_1_proxy.keys():
                    dict_1_proxy[variable] = index[i]
                if variable in dict_2_proxy.keys():
                    dict_2_proxy[variable] = index[i]
                i += 1

            new_table[index] = self._table[tuple(dict_1_proxy.values())] * t2._table[tuple(dict_2_proxy.values())]

        return BeliefTable(new_variables, new_table)

    def divide_table(self, t2):
        """
        Performs BeliefTable division, same idea as multiplication but with division.
        If there's a 0/0 division the result will be 0

        :param t2: The divider
        :type t2: BeliefTable
        :return: the result of the division
        :rtype: BeliefTable
        """
        # Merge all variables
        new_variables = {**self._variables, **t2._variables}

        # Create empty table
        new_shape = (2,) * len(new_variables)
        new_table = np.empty(new_shape)

        for index, value in np.ndenumerate(new_table):
            # Find the corresponding indexes in the 2 multiplying terms
            dict_1_proxy = dict.fromkeys(self._variables)
            dict_2_proxy = dict.fromkeys(t2._variables)
            i = 0
            for variable in list(new_variables.keys()):
                if variable in dict_1_proxy.keys():
                    dict_1_proxy[variable] = index[i]
                if variable in dict_2_proxy.keys():
                    dict_2_proxy[variable] = index[i]
                i += 1

            if self._table[tuple(dict_1_proxy.values())] == 0:
                # Make it so 0/anything is 0(No NaN problems)
                new_table[index] = 0
            else:
                new_table[index] = self._table[tuple(dict_1_proxy.values())] / t2._table[tuple(dict_2_proxy.values())]

        return BeliefTable(new_variables, new_table)

    def marginalize(self, new_variables):
        """
        Marginalizes the BeliefTable on a subset of its variables: t_W =\sum_{V\W} t_V with W\subseteq V
        If I want to marginalize t_AB on A, then I will have t_B as a result, with the following values:
        t_{B=0} = t_{A=0,B=0} + t_{A=1, B=0} ; t_{B=1} = T_{A=0,B=1} + t_{A=1,B=1}

        :param new_variables: set of variables to marginalize on
        :type new_variables: dict[string,None]
        :return: the marginalized table
        :rtype: BeliefTable
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

    def multiply_all(self, value):
        """
        Multiply all of the numpy entries table by the given value

        :type value: int or float
        :return: None
        """
        self._table *= value

    def divide_all(self, value):
        """
        Divide all of the numpy entries table by the given value

        :type value: int or float
        :return: None
        """
        self._table /= value

    def _get_variable_index(self, variable):
        """
        Returns the index of the given variable in the variable "list"(it's still an ordered set)

        :type variable: string
        :return: index of the variable
        :rtype: int
        """
        return list(self._variables.keys()).index(variable)

    def get_prob(self, coordinates):
        """
        Returns the entry of the numpy table at the given coordinates.
        It's used to get the value of a certain configuration of variables, for example if the belief table is t_ABC
        and I want the value of t_{A=0,B=1,C=1}, I have to pass (0,1,1)

        :type coordinates: tuple or int
        :return: the value in the np table
        :rtype: int or float
        """
        return self._table[coordinates]

    def get_prob_dict(self, variables):
        """
        Returns the value of the BeliefTable given a configuration of variables.
        If the belief table is t_ABC and I want the value of t_{A=0,B=1,C=1}, I have to pass {'A':0, 'B':1, 'C':1}

        :param variables: Dict of all variables as keys and values as the value of the single variable
        :type variables: dict[string,int or float]
        :return: the value of the configuration
        :rtype: int or float
        """
        if variables.keys() != self._variables.keys():
            raise AttributeError("The variables in the dictionary were wrong")
        # Match with the right index
        coords = []
        for el in self._variables:
            coords.append(variables[el])

        return self.get_prob(tuple(coords))

    def get_variables(self):
        """
        :return: variables of the BeliefTable
        :rtype: dict[string,None]
        """
        return self._variables

    def set_probability_coord(self, coordinates, value):
        """
        Sets the entry of the numpy table at the given coordinates to a given value

        :type coordinates: tuple
        :type value: int or float
        :return: None
        """
        if len(coordinates) != len(self._variables):
            raise AttributeError("Wrong probability coordinate format")
        self._table[coordinates] = value

    def set_probability_dict(self, variables, value):
        """
        Sets the value of the BeliefTable given a configuration of variables and the value.
        If the belief table is t_ABC and I want the value of t_{A=0,B=1,C=1} to be 0.5,
        I have to pass {'A':0, 'B':1, 'C':1} and 0.5.

        :param variables: Dict of all variables as keys and values as the value of the single variable
        :type variables: dict[string,int or float]
        :type value: int or float
        :return: None
        """
        if variables.keys() != self._variables.keys():
            raise AttributeError("The variables in the dictionary were wrong")
        # Match with the right index
        coords = []
        for el in self._variables:
            coords.append(variables[el])

        self.set_probability_coord(tuple(coords), value)

    def __str__(self):
        """
        Represents the BeliefTable as follows,for each entry of the table:
        A:a,B:b,C:c -> value
        """
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
            full_str += ' -> ' + str(x) + '\n'
            i += 1
        return full_str

    def __copy__(self):
        copied_table = self._table.copy()
        copied_vars = self._variables.copy()

        return BeliefTable(copied_vars, copied_table)

"""
TODO reimplement with multiple values(I hope it won't be hell)
"""
class Variable(object):

    name = ""
    values = {}

    def __init__(self, name, values):
        self.name = name
        self.values = dict.fromkeys(values)

    def get_cardinality(self):
        return len(self.values)