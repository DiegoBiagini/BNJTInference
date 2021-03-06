#
# This file contains the data structures used to represent and work on bayesian nets
#
import copy
from collections import Counter

import numpy as np

import util
from tables import BeliefTable
from tables import Variable


class BayesianNet(object):

    def __init__(self, graph=None):
        """
        Initializes the Bayesian Net with the given graph, or empty if no graph is given

        :type graph: dict[Variable,list]
        """

        """
        The bayesian net, made up of nodes(which are random variables), links between them and tables that represent
        the conditional probabilities between fathers and sons
        """
        if graph is None:
            self._graph = {}
        else:
            self._graph = graph

        """
        Dictionary that contains all variables as keys and tables relative to each RV as values
        """
        self._tables = {}

    def add_variable(self, new_variable):
        """
        Add the given variable to the BN

        :type new_variable: Variable
        :return: None
        """
        if new_variable in self._graph.keys():
            raise AttributeError("Trying to add a variable that is already inside the net")
        else:
            self._graph[new_variable] = []
            self._tables[new_variable] = None

    def add_dependence(self, child, father):
        """
        Adds a link between two nodes in the Bayesian Net(child depends on father)

        :type child: Variable or str
        :type father: Variable or str
        :return: None
        """
        if isinstance(child, str):
            child = self.get_variable_by_name(child)
        if isinstance(father, str):
            father = self.get_variable_by_name(father)

        if father not in self._graph.keys():
            raise AttributeError("Invalid father")
        elif child not in self._graph.keys():
            raise AttributeError("Invalid child")
        else:
            self._graph[father].append(child)

    def add_prob_table(self, variable, table):
        """
        Adds a BeliefTable to a variable

        :type variable: Variable or str
        :type table: BeliefTable
        :return: None
        """
        if isinstance(variable, str):
            variable = self.get_variable_by_name(variable)

        if variable not in self._graph:
            raise AttributeError("Variable not valid")

        # Check if the passed table is correct
        vars_in_table = dict.fromkeys([variable] + self.get_fathers(variable))

        if not(Counter(table.get_variables()) == Counter(vars_in_table)):
            raise AttributeError("Table for the variable is not valid")
        self._tables[variable] = table

    def get_table(self, variable):
        """
        Returns the conditional probability table of a variable in the BN

        :type variable: Variable
        :rtype: BeliefTable
        """
        if variable not in self._tables.keys():
            raise AttributeError("Variable not valid")
        return self._tables[variable]

    def get_fathers(self, child):
        """
        Returns all the father variables of a given variable

        :type child: Variable
        :rtype: list[Variables]
        """
        if child not in self._graph.keys():
            raise AttributeError("Child not found")

        fathers = []
        for el in self._graph:
            if child in self._graph[el]:
                fathers.append(el)

        return fathers

    def get_U_probability_string(self):
        """
        Traverse the graph and return a string with how P(U) is calculated according to the BN topology
        e.g. Variables: A,B,C ; A is the father of both B and C; the U-prob string is P(A) P(B|A) P(C|A)

        :return: u-prob
        :rtype: string
        """
        u_prob_string = ""
        for el in self._graph:
            fathers = self.get_fathers(el)
            if fathers == []:
                u_prob_string += "P(" + str(el) + ") "
            else:
                u_prob_string += "P(" + str(el) + "|"

                for dad in fathers:
                    u_prob_string += str(dad) + ","
                u_prob_string = u_prob_string[:-1]

                u_prob_string += ") "

        return u_prob_string

    def get_variable_by_name(self, name):
        """
        Searches for a variable with the given name in the bayesian net

        :type name: str
        :rtype: Variable
        """
        for var in self._tables.keys():
            if var.name == name:
                return var

        raise AttributeError("Variable not found")

    def get_variables(self):
        """
        :rtype: dict[Variable, None]
        """
        return dict.fromkeys(self._graph.keys())

    def get_graph(self):
        """
        :return: dict[Variable,[Variable]]
        """
        return self._graph

    def __str__(self):
        """
        Returns a string containing the variables of the BN and its structure, that is {father : [child1, child2], ...}
        """
        rstring = 'Variables:' + str([str(var) for var in self._graph.keys()]) + '\n {\n'
        for father in self._graph.keys():
            if len(self._graph[father]) > 0:
                rstring += "\t" + father.name + " : ["

                for son in self._graph[father]:
                    rstring += son.name + ", "
                rstring = rstring[:-2]

                rstring += "],\n"
        rstring = rstring[:-2] + "\n}"
        return rstring


class JunctionTree(object):
    """
    Class that represents the junction tree relative to a bayesian net, made up of cliques and separators and their
    connections
    """

    def __init__(self, variables):
        """
        Initialize the JunctionTree with the given variables and with no cliques/separators

        :type variables: dict[Variable,None] or list[Variable]
        """
        """
        Dict of Variables, contains the variables of the bayesian net and by extension the variables of the junction tree
        """
        self._variables = dict.fromkeys(variables)

        """
        Both are a list of Node elements
        """
        self._cliques = []
        self._separators = []

        """
        Dictionary that contains all variables as keys and a reference to a clique as values. 
        For each variable A a clique that contains pa(A) \cup {A} is chosen during the Junction Tree creation, this dict 
        stores this information
        """
        self._chosen_clique = {}

    def add_clique(self, clique):
        """
        Add a valid clique made up of variables to the list of cliques
        Variables can be both list/dict of references or a list of variable names

        :type clique: list[str] or dict[Variable,None] or list[Variable]
        :return: None
        """
        if all(isinstance(x, str) for x in clique):
            clique = [self.get_variable_by_name(name) for name in clique]

        clique = dict.fromkeys(clique)
        if not clique.keys() <= self._variables.keys():
            raise AttributeError("The given clique is not valid for the junction tree")
        new_node = Node(BeliefTable(clique))
        self._cliques.append(new_node)

    def add_separator(self, separator):
        """
        Add a valid separator made up of variables to the list of separators
        Variables can be both list/dict of references or a list of variable names

        :type separator: list[str] or dict[Variable,None] or list[Variable]
        :return: None
        """
        if all(isinstance(x, str) for x in separator):
            separator = [self.get_variable_by_name(name) for name in separator]

        separator = dict.fromkeys(separator)
        if not separator.keys() <= self._variables.keys():
            raise AttributeError("The given separator is not valid for the junction tree")
        new_node = Node(BeliefTable(separator))
        self._separators.append(new_node)

    def add_link(self, clique, separator):
        """
        Adds a link between a clique and a separator, given two dicts/lists to represent the Nodes

        :type clique: list[str] or dict[Variable,None] or list[Variable]
        :type separator: list[str] or dict[Variable,None] or list[Variable]
        :return: None
        """
        if all(isinstance(x, str) for x in clique):
            clique = [self.get_variable_by_name(name) for name in clique]
        if all(isinstance(x, str) for x in separator):
            separator = [self.get_variable_by_name(name) for name in separator]

        true_clique = self.get_clique(clique)
        true_sep = self.get_separator(separator)
        self._add_link(true_clique, true_sep)

    def _add_link(self, clique, separator):
        """
        Adds a link between a clique and a separator, given references to both Nodes

        :type clique: Node
        :type separator: Node
        :return: None
        """
        if clique not in self._cliques or separator not in self._separators:
            raise AttributeError("Clique or separator not valid for the junction tree")
        if not(separator.get_variables().keys() <= clique.get_variables().keys()):
            raise AttributeError("Clique does not contain the variables in the separator")

        clique.add_neighbour(separator)
        separator.add_neighbour(clique)

    def connect_cliques(self, clique1, clique2):
        """
        Connect two neighbouring cliques by creating a separator between them and linking them all

        :type clique1: list[str] or list[Variable]
        :param clique2: list[str] or list[Variable]
        :return:
        """
        if all(isinstance(x, str) for x in clique1):
            clique1 = [self.get_variable_by_name(name) for name in clique1]
        if all(isinstance(x, str) for x in clique2):
            clique2 = [self.get_variable_by_name(name) for name in clique2]

        clique_variables_list = [node.get_variables() for node in self._cliques]
        if dict.fromkeys(clique1) not in clique_variables_list or dict.fromkeys(clique2) not in clique_variables_list:
            raise AttributeError("One of the cliques wasn't valid")

        common_vars = [var for var in clique1 if var in clique2]
        if len(common_vars) == 0:
            raise AttributeError("The cliques aren't neighbouring")

        self.add_separator(common_vars)
        self.add_link(clique1, common_vars)
        self.add_link(clique2, common_vars)

    def set_variable_chosen_clique(self, variable, clique):
        """
        Sets the clique(passed as a dict/list) chosen during creation for the given variable

        :type variable: Variable or str
        :type clique: list[str] or dict[Variable,None] or list[Variable]
        :return: None
        """
        if isinstance(variable, str):
            variable = self.get_variable_by_name(variable)
        if all(isinstance(x, str) for x in clique):
            clique = [self.get_variable_by_name(name) for name in clique]

        true_clique = self.get_clique(clique)
        self._set_variable_chosen_clique(variable, true_clique)

    def _set_variable_chosen_clique(self, variable, clique):
        """
        Sets the clique(passed as a reference) chosen during creation for the given variable

        :type variable: Variable
        :type clique: Node
        :return: None
        """
        if variable not in self._variables.keys():
            raise AttributeError("Variable not valid")

        if clique not in self._cliques:
            raise AttributeError("Clique not valid")

        self._chosen_clique[variable] = clique

    def get_clique(self, clique_vars):
        """
        Returns a reference to a clique, given the variables it's made up of

        :type clique_vars: list[str] or dict[Variable,None] or list[Variable]
        :return: a reference to the clique if it's found
        :rtype: Node
        """
        if all(isinstance(x, str) for x in clique_vars):
            clique_vars = [self.get_variable_by_name(name) for name in clique_vars]

        dict_clique = dict.fromkeys(clique_vars)

        for clique in self._cliques:
            if clique.get_variables() == dict_clique:
                return clique

        raise AttributeError("Clique not valid")

    def get_separator(self, separator_vars):
        """
        Returns a reference to a separator, given the variables it's made up of

        :type separator_vars: list[str] or dict[Variable,None] or list[Variable]
        :return: a reference to the separator if it's found
        :rtype: Node
        """
        if all(isinstance(x, str) for x in separator_vars):
            separator_vars = [self.get_variable_by_name(name) for name in separator_vars]

        dict_sep = dict.fromkeys(separator_vars)

        for sep in self._separators:
            if sep.get_variables() == dict_sep:
                return sep

        raise AttributeError("Separator not valid")

    def add_evidence(self, variable, value):
        """
        Add evidence to a variable in the JunctionTree by setting the table of the relative clique to the correct
        configuration(according to evidence)

        :type variable: Variable or str
        :param value: evidence to assert, can only be a correct value according to the variable
        :type value: int or string
        :return: None
        """
        if isinstance(variable, str):
            variable = self.get_variable_by_name(variable)
        if isinstance(value, str):
            value = value.lower()

        if variable not in self._variables:
            raise AttributeError("Variable not valid")
        if not variable.is_valid(value):
            raise AttributeError("Value not valid")

        # Find the clique to update
        chosen_clique = self._chosen_clique[variable]
        chosen_clique.received_evidence = True

        table = copy.copy(chosen_clique.get_prob_table())

        # Select which cell in the table has to be set to 0(those that contradict the evidence)
        confuted_values = [x for x in variable.values if x != value]
        for conf_value in confuted_values:
            coord_dict = {}
            for el in table.get_variables():
                if el != variable:
                    coord_dict[el.name] = slice(None)
                else:
                    coord_dict[el.name] = conf_value

            table.set_probability_dict(coord_dict, 0)

        # Calculate P(U |e) = P(U,e)/P(e)
        try:
            table.divide_all(table.marginalize([variable]).get_prob_dict({variable.name: value}))
            chosen_clique.set_prob_table(table)
        except RuntimeWarning:
            raise RuntimeError("Conflicting evidence was entered")

    def get_joint_probability_table(self):
        """
        Returns the joint probability table of the whole Bayesian Net by multiplying all the tables of the cliques and
        dividing the result by all the tables of the separators

        :return: joint probability table
        :rtype: BeliefTable
        """
        # Multiply all the tables of the cliques
        result_table = BeliefTable(self._variables, np.ones(util.get_shape_from_var_dict(self._variables)))
        for clique in self._cliques:
            result_table = result_table.multiply_table(clique.get_prob_table())

        # Divide the result by all the separator tables
        for sep in self._separators:
            result_table = result_table.divide_table(sep.get_prob_table())

        return result_table

    def calculate_variable_probability_on_universe(self, variable):
        """
        Calculate the probabilities of one variable in the given configuration of the JunctionTree by
        calculating the joint prob table and then marginalizing on the given variable.
        Does not use the efficiency of Hugin propagation.

        :type variable: Variable or str
        :return: probability table of the given variable
        :rtype: BeliefTable
        """
        if isinstance(variable, str):
            variable = self.get_variable_by_name(variable)

        variable_dict = {variable: None}

        if not variable_dict.keys() <= self._variables.keys():
            raise AttributeError("Variable not valid")

        table = self.get_joint_probability_table()

        marginalized_table = table.marginalize(variable_dict)
        norm_constant = 0
        i = 0
        for value in variable.values:
            norm_constant += marginalized_table.get_prob(i)
            i += 1

        marginalized_table.divide_all(norm_constant)
        return marginalized_table

    def calculate_variable_probability(self, variable):
        """
        Calculate the probabilities of the given variable by consulting the clique chosen for that variable during the
        JunctionTree creation. Needs Hugin propagation to give out correct results.

        :type variable: Variable or str
        :return: table over the single variable
        :rtype: BeliefTable
        """
        if isinstance(variable, str):
            variable = self.get_variable_by_name(variable)

        if variable not in self._variables:
            raise AttributeError("Variable not valid")
        chosen_clique = self._chosen_clique[variable]
        return chosen_clique.get_prob_table().marginalize({variable: None})

    @staticmethod
    def absorption(first, separator, second):
        """
        The basic operation for message passing between the JunctionTree,executed on two cliques and a common separator.
        second absorbs information from first

        :param first: V, clique that sends information
        :type first: Node
        :param separator: S, common separator of V and W
        :type separator: Node
        :param second: W, clique that receives information
        :type second: Node
        :return: None
        """
        if (separator not in first.get_neighbours() or separator not in second.get_neighbours()
                or first not in separator.get_neighbours() or second not in separator.get_neighbours()):
            raise AttributeError("Combination of nodes not valid")
        tv = first.get_prob_table()
        ts = separator.get_prob_table()
        tw = second.get_prob_table()

        ts_star = tv.marginalize(separator.get_variables())
        separator.set_prob_table(ts_star)

        second.set_prob_table(tw.multiply_table(ts_star.divide_table(ts)))

    def distribute_evidence(self, node):
        """
        Second main operation of Hugin propagation. A node(initially the root node) sends all its neighbours the
        information it collected during previous CollectEvidence or DistributeEvidence, this is done recursively.

        :param node: the Node that sends its neighbours information
        :type node: Node
        :return: None
        """
        if node not in self._cliques:
            raise AttributeError("Wrong starting clique")

        # Keep track of visited nodes
        visited_labels = dict.fromkeys(self._cliques, False)

        # Perform a pseudo-BF traversal
        queue = []
        visited_labels[node] = True
        queue.append(node)

        while len(queue) != 0:
            v = queue.pop(0)

            for neighbour in self.get_neighbouring_cliques(v):
                if not visited_labels[neighbour]:
                    visited_labels[neighbour] = True
                    queue.append(neighbour)

                    common_separator = [x for x in v.get_neighbours() if x in neighbour.get_neighbours()][0]
                    JunctionTree.absorption(v, common_separator, neighbour)

    def collect_evidence(self, node):
        """
        First main operation of Hugin propagation.
        A node asks all its neighbours to send it evidence, if they are not allowed to do so (they haven't received
        evidence either) they pass the request to all their neighbouring nodes, except to the one that sent it

        :param node: the node that asks its neighbours for evidence
        :type node: Node
        :return: None
        """
        if node not in self._cliques:
            raise AttributeError("Wrong starting clique")

        # Keep track of visited nodes
        visited_labels = dict.fromkeys(self._cliques, False)

        # Keep track of traversal to send back evidence to the correct node
        parents = dict.fromkeys(self._cliques, None)

        # Perform a pseudo-BF traversal
        queue = []
        visited_labels[node] = True
        queue.append(node)

        while len(queue) != 0:
            v = queue.pop(0)

            # If a Node with evidence is found propagate its information over all its ancestors
            if v.received_evidence:
                current_child = v
                ancestors = parents[current_child]
                while ancestors is not None:
                    common_separator = [x for x in current_child.get_neighbours() if x in ancestors.get_neighbours()][0]

                    JunctionTree.absorption(current_child, common_separator, ancestors)

                    current_child = ancestors
                    ancestors = parents[current_child]

                # Update the state of evidence collecting
                v.received_evidence = False

            for neighbour in self.get_neighbouring_cliques(v):
                if not visited_labels[neighbour]:
                    visited_labels[neighbour] = True
                    parents[neighbour] = v
                    queue.append(neighbour)

    def sum_propagate(self):
        """
        Operation that propagates the evidence over all the JunctionTree by calling CollectEvidence, DistributeEvidence
        and then normalizing the tables of all cliques/separator
        """
        # Choose a root(any one should be fine)
        root = self._cliques[0]

        self.collect_evidence(root)
        self.distribute_evidence(root)

        # Normalize
        # Find normalizing constant by marginalizing on any variable
        variable_chosen = list(self._variables)[0]
        norm_table = self.calculate_variable_probability(variable_chosen)

        norm_constant = 0
        for i in range(len(variable_chosen.values)):
            norm_constant += norm_table.get_prob(i)

        for node in self._cliques + self._separators:
            node.get_prob_table().divide_all(norm_constant)

    def get_neighbouring_cliques(self, clique):
        """
        Returns the neighbouring cliques of a given clique, skips separators

        :type clique: Node
        :return: neighbouring cliques
        :rtype: list[Node]
        """
        if clique not in self._cliques:
            raise AttributeError("Clique not valid")

        neigh_cliques = []
        for sep in clique.get_neighbours():
            for clique_of_sep in sep.get_neighbours():
                if clique_of_sep not in neigh_cliques and clique_of_sep != clique:
                    neigh_cliques.append(clique_of_sep)

        return neigh_cliques

    def initialize_tables(self, bayes_net):
        """
        Initializes the JunctionTree to the initial values of the tables of a BayesianNet and distributes their
        information over all the Nodes

        :type bayes_net: BayesianNet
        :return: None
        """
        # Check if the variables are the same
        if self._variables != bayes_net.get_variables():
            raise AttributeError("The variables in the BayesianNet and in the JunctionTree aren't the same")
        # Set all separators and cliques to 1
        for node in self._separators + self._cliques:
            node.get_prob_table().set_probability_coord((slice(None),) * len(node.get_variables()), 1)

        # Choose the correct clique for each variable and store its table in it
        for variable in self._variables:
            clique_for_var = [variable] + bayes_net.get_fathers(variable)

            # Find the first clique that contains a variable and its fathers
            clique_ref = None
            for clique in self._cliques:
                if all(v in list(clique.get_variables().keys()) for v in clique_for_var):
                    clique_ref = clique
            if clique_ref is None:
                raise AttributeError(str(clique_for_var) + " wasn't found")

            self._chosen_clique[variable] = clique_ref

            table = bayes_net.get_table(variable)
            clique_ref.set_prob_table(clique_ref.get_prob_table().multiply_table(table))

        # Distribute initial information by allowing a round of message passing
        for node in self._cliques:
            if node in self._chosen_clique.values():
                node.received_evidence = True

        self.sum_propagate()

    def get_clique_from_dict(self, clique):
        """
        Get reference to a clique given the variables it's made up of

        :type clique: dict[Variable,None] or list[Variable]
        :return: the clique or None if it doesn't exists
        :rtype: Node
        """
        clique = dict.fromkeys(clique)
        for element in self._cliques:
            if element.get_variables() == clique:
                return element
        return None

    def get_separator_from_dict(self, sep):
        """
        Get reference to a separator given the variables it's made up of

        :type sep: dict[Variable,None] or list[Variable]
        :return: the separator or None if it doesn't exists
        :rtype: Node
        """
        sep = dict.fromkeys(sep)
        for element in self._separators:
            if element.get_variables() == sep:
                return element
        return None

    def get_variable_by_name(self, name):
        """
        Searches for a variable with the given name in the JunctionTree

        :type name: str
        :rtype: Variable
        """
        for var in self._variables:
            if name == var.name:
                return var
        raise AttributeError("Variable not found")

    def get_variables(self):
        """
        :rtype:dict[Variable]
        :return: A dictionary containing the variables
        """
        return self._variables

    def get_cliques_and_seps(self):
        """
        :rtype: tuple[list[Node],list[Node]]
        :return: A tuple containing the list of cliques and separators
        """
        return self._cliques, self._separators

    def __str__(self):
        """
        Represents the JunctionTree by listing all variables, all cliques, all separators and which clique was chosen
        for each variable
        """
        rstring = 'Variables:' + str([str(var) for var in self._variables]) + '\n'

        rstring += 'Cliques:\n'
        for clique in self._cliques:
            rstring += clique.node_vars_to_string() + ': '
            for neigh in clique.get_neighbours():
                rstring += neigh.node_vars_to_string() + ','
            rstring = rstring[:-1] + '\n'

        rstring += 'Separators:\n'
        for sep in self._separators:
            rstring += sep.node_vars_to_string() + ': '
            for neigh in sep.get_neighbours():
                rstring += neigh.node_vars_to_string() + ','
            rstring = rstring[:-1] + '\n'

        rstring += 'Chosen clique for each variable:\n'
        for el in self._chosen_clique.keys():
            rstring += el.name + ':' + self._chosen_clique[el].node_vars_to_string() + ', '
        rstring = rstring[:-2]

        return rstring


class Node(object):
    """
    Data structure that represents a clique or a separator in a JunctionTree
    """
    def __init__(self, table):
        """
        Initialize a Node to a given table,with no neighbours and with no evidence given

        :type table: BeliefTable
        """
        """
        BeliefTable of the Node, contains the variables that make up the clique/separator as well
        """
        self._table = table

        """
        List of other Nodes that represents the neighbours of the clique/separator
        """
        self._neighbours = []

        """
        Boolean value that represents if the Node has received evidence, used in propagation
        """
        self.received_evidence = False

    def get_variables(self):
        """
        :return: variables
        :rtype: dict[Variable, None]
        """
        return self._table.get_variables()

    def get_prob_table(self):
        """
        :return: table
        :rtype: BeliefTable
        """
        return self._table

    def set_prob_table(self, table):
        """
        :type table: BeliefTable
        :return: None
        """
        self._table = table

    def add_neighbour(self, node):
        """
        :type node: Node
        :return: None
        """
        self._neighbours.append(node)

    def get_neighbours(self):
        """
        :return: neighbours
        :rtype: list[Node]
        """
        return self._neighbours

    def node_vars_to_string(self):
        """
        Returns a string that contains all variables of the Node, separated by single dots; used for representation
        purposes in JunctionTree

        :rtype: string
        """
        return '.'.join([x.name for x in list(self.get_variables())])
