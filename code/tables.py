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
    A dictionary of variables,used as an ordered set. Each variable is an object of type Variable in the dictionary 
    with None as its value.
    """
    _table = None
    """
    A numpy table that contains all the entries of the BeliefTable. 
    # For simplicity each variable can only take True or False as its values, so the size of the table is always 2^n, #
    The size of the table is \prod_{v\in V}|v| where V is the set of variables , that is the product of the number of
    values each variable can take
    """

    def __init__(self, variables, table=None):
        """
        Initializes the BeliefTable with the variables and optionally their table, if no table is given an appropriate
        table of zeros will be used

        :type table: np.ndarray
        :type variables: list[Variable] or dict[Variable,None]
        """
        self._variables = dict.fromkeys(variables)
        if table is not None:
            self._table = table

            # Check table size
            if self.get_vars_size() != self._table.size:
                raise AttributeError("Wrong array size")
        else:
            self._table = np.zeros(util.get_shape_from_var_dict(self._variables))

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
        new_shape = util.get_shape_from_var_dict(new_variables)
        new_table = np.empty(new_shape)

        for index, value in np.ndenumerate(new_table):
            # Find the corresponding indexes in the 2 multiplying terms
            # For example: t_{A,B}*t_{A,C}=t_{A,B,C}
            # the element in position (a0,b0,c1) in the result is the product of the elements in positions
            # (a0,b0) and in position (b0,c1)
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
        new_shape = util.get_shape_from_var_dict(new_variables)
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
        :type new_variables: dict[Variable,None] or list[Variable]
        :return: the marginalized table
        :rtype: BeliefTable
        """
        new_variables = dict.fromkeys(new_variables)
        if not (new_variables.keys() < self._variables.keys()):
            raise AttributeError("Variables to marginalize on must be a subset of variables of the table")

        # new_table = np.zeros((2,)*len(new_variables))
        new_shape = []
        for el in new_variables.keys():
            new_shape.append(el.get_cardinality())

        new_table = np.zeros(tuple(new_shape))

        sum_variables = util.subtract_ordered_dict(self._variables, new_variables)
        sum_shape = []
        for el in sum_variables.keys():
            sum_shape.append(el.get_cardinality())
        sum_entries = util.shape_to_list_of_entries(sum_shape)

        # Create template for indexing, setting up which variables have to be extracted whole :
        # Marginalizing on AB over t_ABC means summing (:,:,c0) + (:,:,c1) +... (that is over C values)
        coord_template = []
        for el in self._variables:
            if el in new_variables:
                coord_template.append(slice(None))
            else:
                coord_template.append(0)

        # Iterate over all possible values of V\W
        for entry in sum_entries:
            #bin_string = format(i, '0' + str(len(sum_variables)) + 'b')
            actual_coord = coord_template.copy()

            # Find the subtable by substituting the non ':' places in the index template with actual indexes
            index = 0     # Index for the binary string
            for ind, element in enumerate(actual_coord):
                if element != slice(None):
                    actual_coord[ind] = entry[index]
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

        :type variable: Variable
        :return: index of the variable
        :rtype: int
        """
        return list(self._variables.keys()).index(variable)

    def get_prob(self, coordinates):
        """
        Returns the entry of the numpy table at the given coordinates.
        Don't use this to get values of the table when the number values of any variable is more than 2

        :type coordinates: tuple or int
        :return: the value in the np table
        :rtype: int or float
        """
        return self._table[coordinates]

    def get_prob_var_values(self, values_coordinates):
        """
        It's used to get the value of a certain configuration of variables, for example if the belief table is t_ABC
        and I want the value of t_{A='a1',B='b0',C='c1'}, I have to pass ('a1','b0','c1')
        Value coordinates must be in the correct order, don't use this if you don't know the true order of the variables

        :param values_coordinates: a tuple of var values
        :type values_coordinates: tuple[string or int]
        :return: the value in the np table
        :rtype: int or float
        """
        if len(values_coordinates) != len(self._variables):
            raise AttributeError("Wrong number of variables")

        vars_list = list(self._variables.keys())

        values_coordinates = (v.lower() if isinstance(v, str) else v for v in values_coordinates)

        actual_coordinates = []
        i = 0
        for el in values_coordinates:
            actual_coordinates.append(vars_list[i].get_value_index(el))
            i += 1
        return self.get_prob(tuple(actual_coordinates))

    def get_prob_dict(self, vars_and_vals):
        """
        Returns the value of the BeliefTable given a configuration of variables.
        If the belief table is t_ABC and I want the value of t_{A='a1',B='b1',C='c1'}, I have to pass {'A':'a1', '
        B':'b1', 'C':'c1'} , where 'A','B' and 'C' are the names of the variables

        :param vars_and_vals: Dict of all variable names as keys and values as the value of the single variable
        :type vars_and_vals: dict[string,int or string]
        :return: the value of the configuration
        :rtype: int or float
        """
        var_names = self.get_variable_names()
        if sorted(vars_and_vals.keys()) != sorted(var_names):
            raise AttributeError("The variable names in the dictionary were wrong")
        # Set to lowercase
        vars_and_vals = {v: w.lower() if isinstance(w, str) else w for v, w in vars_and_vals.items() if isinstance(w, str)}

        # Match with the right index
        coords = []
        for el in var_names:
            coords.append(vars_and_vals[el])

        return self.get_prob_var_values(tuple(coords))

    def get_variables(self):
        """
        :return: variables of the BeliefTable
        :rtype: dict[string,None]
        """
        return self._variables

    def get_variable_names(self):
        """
        Returns a list containing the names of the variables of the table

        :rtype: list[string]
        """
        names = []
        for el in self._variables:
            names.append(el.name)
        return names

    def set_probability_coord(self, coordinates, value):
        """
        Sets the entry of the numpy table at the given coordinates to a given value
        Don't use this to get values of the table when the number values of any variable is more than 2

        :type coordinates: tuple
        :type value: int or float
        :return: None
        """
        if len(coordinates) != len(self._variables):
            raise AttributeError("Wrong probability coordinate format")
        self._table[coordinates] = value

    def set_probability_var_values(self, values_coordinates, prob_value):
        """
        Sets the entry of the numpy table at the given coordinates,passed as a tuple of var values, to a given value
        The values must be in order

        :param values_coordinates: tuple of consistent variable values
        :type values_coordinates: tuple[int or string]
        :param prob_value: Value to enter into the table
        :type prob_value: float or int
        :return: None
        """
        if len(values_coordinates) != len(self._variables):
            raise AttributeError("Wrong number of variables")

        values_coordinates = (v.lower() if isinstance(v, str) else v for v in values_coordinates)

        vars_list = list(self._variables.keys())

        actual_coordinates = []
        i = 0
        for el in values_coordinates:
            if isinstance(el, slice):
                actual_coordinates.append(el)
            else:
                actual_coordinates.append(vars_list[i].get_value_index(el))
            i += 1

        return self.set_probability_coord(tuple(actual_coordinates), prob_value)

    def set_probability_dict(self, vars_and_vals, value):
        """
        Sets the value of the BeliefTable given a configuration of variables and the value.
        If the belief table is t_ABC and I want the value of t_{A='a1',B='b1',C='c1'} to be 0.5,
        I have to pass {'A':'a1', 'B':'b1', 'C':'c1'} and 0.5, if 'A','B' and 'C' are the names of the variables.

        :param vars_and_vals: Dict of all variables names as keys and values as the value of the single variable
        :type vars_and_vals: dict[string,int or string]
        :type value: int or float
        :return: None
        """
        var_names = self.get_variable_names()
        if sorted(vars_and_vals.keys()) != sorted(var_names):
            raise AttributeError("The variable names in the dictionary were wrong")

        # Set to lowercase
        vars_and_vals = {v: w.lower() if isinstance(w, str) else w for v, w in vars_and_vals.items() }

        # Match with the right index
        coords = []
        for el in var_names:
            coords.append(vars_and_vals[el])

        return self.set_probability_var_values(tuple(coords), value)

    def get_vars_size(self):
        """
        Calculate the size of the table according to the values the variables of the table can take

        :return: size of the table
        :rtype: int
        """
        size = 1
        for el in self._variables.keys():
            size *= el.get_cardinality()

        return size

    def __str__(self):
        """
        Represents the BeliefTable as follows,for each entry of the table:
        A:'a',B:'b',C:'c' -> value
        """
        full_str = ''

        # i = 0
        for idx, x in np.ndenumerate(self._table):

            i = 0
            for val in self._variables:
                full_str += val.name + ":"
                value = list(val.values)[idx[i]]
                if isinstance(value, str):
                    full_str += "'" + value + "', "
                else:
                    full_str += str(value) + ", "
                i += 1
            full_str += "-> " + "{:.4f}".format(x) + "\n"

        return full_str

    def __copy__(self):
        copied_table = self._table.copy()
        copied_vars = self._variables.copy()

        return BeliefTable(copied_vars, copied_table)


class Variable(object):
    """
    Class that represents a variable and the values it can take
    """
    name = ""
    label = ""
    """
    Describes the variable's purpose, two Variables with the same name and values but different label are still the same
    """
    values = {}
    """
    Values the variable can take, they can be int or strings but not both. If they are strings they are case insensitive
    """

    def __init__(self, name, label, values):
        """
        Initializes a variable with the given name,label and values. Only use alphanumeric values for the name and values
        if you don't want to tempt fate.

        :type name: string
        :type label: string
        :type values: list[string or int]
        """
        self.name = name
        self.label = label

        if all(isinstance(x, str) for x in values):
            self.values = dict.fromkeys([x.lower() for x in values])
        else:
            self.values = dict.fromkeys(values)

        # Check if all values are of the same type
        value_list = list(self.values.keys())
        first_type = type(value_list[0])
        if not all(isinstance(x, first_type) for x in value_list):
            raise AttributeError("The value list can't contain  both integers and strings")

    def get_cardinality(self):
        """
        Returns the number of values a variable can take

        :rtype: int
        """
        return len(self.values)

    def get_value_index(self, value):
        """
        Get the index of the value in the set of values

        :type value: int or string
        :return: index
        :rtype: int
        """
        if isinstance(value, str):
            value = value.lower()
        if value not in self.values:
            raise AttributeError("Value not valid for the given variable")
        return list(self.values.keys()).index(value)

    def is_valid(self, value):
        """
        Checks if the given value is valid for the variable.

        :type value: str or int
        :rtype: bool
        """
        if isinstance(value, str):
            value = value.lower()

        return value in self.values

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return AttributeError("Wrong comparison types")

        return self.name == other.name and self.values == other.values

    def __repr__(self):
        stringed_var = self.name + ":"
        for val in sorted(list(self.values.keys())):
            # Differentiate between string values and int values
            if isinstance(val, str):
                stringed_var += "'" + val + "'"
            else:
                stringed_var += str(val)
            stringed_var += ','

        return stringed_var[:-1]

    def __str__(self):
        stringed_var = self.name + "(" + self.label + ") :"
        for val in sorted(list(self.values.keys())):
            # Differentiate between string values and int values
            if isinstance(val, str):
                stringed_var += "'" + val + "'"
            else:
                stringed_var += str(val)
            stringed_var += ','

        return stringed_var[:-1]

    def __copy__(self):
        copied_name = str(self.name)
        copied_label = str(self.label)
        copied_values = self.values.copy()

        return Variable(copied_name, copied_label, copied_values)

    def __hash__(self):
        return hash(repr(self))
