#
# This file contains the data structures used to represent and work on bayesian nets
#
import numpy as np

from tables import BeliefTable


class BayesianNet(object):
    """
    A bayesian net, made up of nodes(which are random variables), links between them and tables that represent
    the conditional probabilities between fathers and sons
    """

    _graph = {}
    """
    Dictionary that represents the structure of the bayesian net, the keys are the variables of the BN(strings), 
    the values are lists of other variables(the sons of the key)
    """
    _tables = {}
    """
    Dictionary that contains all variables as keys and tables relative to each RV as values
    """

    def __init__(self, graph=None):
        """
        Initializes the Bayesian Net with the given graph, or empty if no graph is given

        :type graph: dict[string,list]
        """
        if graph is None:
            self._graph = {}
        else:
            self._graph = graph

        self._tables = {}

    def add_variable(self, new_variable):
        """
        Add the given variable to the BN

        :param new_variable: variable to add
        :type new_variable: string
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

        :param child: child node
        :type child: string
        :param father: father node
        :type father: string
        :return: None
        """
        if father not in self._graph.keys():
            raise AttributeError("Invalid father")
        elif child not in self._graph.keys():
            raise AttributeError("Invalid child")
        else:
            self._graph[father].append(child)

    def add_prob_table(self, variable, table):
        """
        Adds a Belief Table to a variable

        :type variable: string
        :type table: BeliefTable
        :return: None
        """
        if variable not in self._graph:
            raise AttributeError("Variable not valid")

        # Check if the passed table is correct
        vars_in_table = [variable] + self.get_fathers(variable)

        if not(sorted(table.get_variables()) == sorted(vars_in_table)):
            raise AttributeError("Table for the variable is not valid")
        self._tables[variable] = table

    def get_table(self, variable):
        """
        Returns the conditional probability table of a variable in the BN

        :type variable: string
        :return: table relative to the variable
        :rtype: BeliefTable
        """
        if variable not in self._tables.keys():
            raise AttributeError("Variable not valid")
        return self._tables[variable]

    def get_fathers(self, child):
        """
        Returns all the father variables of a given variable

        :type child: string
        :return: fathers
        :rtype: list[string]
        """
        if child not in self._graph.keys():
            raise AttributeError("Child not found")

        fathers = []
        for el in self._graph:
            if child in self._graph[el]:
                fathers.append(el)

        return fathers

    def is_acyclic(self):
        """
        Checks if the Bayesian Net is acyclic( which it should be if it's a BN) using topological ordering
        https://www.cs.hmc.edu/~keller/courses/cs60/s98/examples/acyclic/

        :rtype: bool
        """

        proxy_graph = self._graph.copy()

        while len(proxy_graph) != 0:
            # Find any leaf
            leaf = None
            for el in proxy_graph:
                if len(proxy_graph[el]) == 0:
                    leaf = el
                    break
            # If no leaves were found the graph is cyclic
            if leaf is None:
                return False

            # Remove the leaf found and all the arcs going into it
            del proxy_graph[leaf]

            for el in proxy_graph:
                proxy_graph[el] = [x for x in proxy_graph[el] if x != leaf]

        # If the graph has no nodes it's acyclic
        return True

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

    def __str__(self):
        """
        Represents the BN as only its graph structure, that is {father : [child1, child2], ...}
        """
        return str(self._graph)


class JunctionTree(object):
    """
    Class that represents the junction tree relative to a bayesian net, made up of cliques and separators and their
    connections
    """

    _variables = {}
    """
    Dict of strings, contains the variables of the bayesian net and by extension the variables of the junction tree
    """

    _cliques = []
    _separators = []
    """
    Both are a list of Node elements
    """

    _chosen_clique = {}
    """
    Dictionary that contains all variables as keys and a reference to a clique as values. 
    For each variable A a clique that contains pa(A) \cup {A} is chosen during the Junction Tree creation, this dict 
    stores this information
    """

    def __init__(self, variables):
        """
        Initialize the JunctionTree with the given variables and with no cliques/separators

        :type variables: dict[string,None] or list[str]
        """
        self._variables = dict.fromkeys(variables)
        self._cliques = []
        self._separators = []
        self._chosen_clique = {}

    def add_clique(self, clique):
        """
        Add a valid clique made up of variables to the list of cliques

        :type clique: dict[string,None] or list[string]
        :return: None
        """
        clique = dict.fromkeys(clique)
        if not clique.keys() <= self._variables.keys():
            raise AttributeError("The given clique is not valid for the junction tree")
        new_node = Node(BeliefTable(clique))
        self._cliques.append(new_node)

    def add_separator(self, separator):
        """
        Add a valid separator made up of variables to the list of separators

        :type separator: dict[string,None] or list[string]
        :return: None
        """
        separator = dict.fromkeys(separator)
        if not separator.keys() <= self._variables.keys():
            raise AttributeError("The given separator is not valid for the junction tree")
        new_node = Node(BeliefTable(separator))
        self._separators.append(new_node)

    def add_link(self, clique, separator):
        """
        Adds a link between a clique and a separator, given two dicts/lists to represent the Nodes

        :type clique: dict[string,None] or list[str]
        :type separator: dict[string,None] or list[str]
        :return: None
        """
        true_clique = self._get_clique(clique)
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

    def set_variable_chosen_clique(self, variable, clique):
        """
        Sets the clique(passed as a dict/list) chosen during creation for the given variable

        :type variable: string
        :type clique: dict[string,None] or list[str]
        :return: None
        """
        true_clique = self._get_clique(clique)
        self._set_variable_chosen_clique(variable, true_clique)

    def _set_variable_chosen_clique(self, variable, clique):
        """
        Sets the clique(passed as a reference) chosen during creation for the given variable

        :type variable: string
        :type clique: Node
        :return: None
        """
        if variable not in self._variables.keys():
            raise AttributeError("Variable not valid")

        if clique not in self._cliques:
            raise AttributeError("Clique not valid")

        self._chosen_clique[variable] = clique

    def _get_clique(self, clique_vars):
        """
        Returns a reference to a clique, given the variables it's made up of

        :type clique_vars: dict[string,None] or list[string]
        :return: a reference to the clique if it's found
        :rtype: Node
        """
        dict_clique = dict.fromkeys(clique_vars)
        found_clique = None
        for element in self._cliques:
            if element.get_variables() == dict_clique:
                found_clique = element
                break

        if found_clique is None:
            raise AttributeError("Clique not valid")
        return found_clique

    def get_separator(self, separator_vars):
        """
        Returns a reference to a separator, given the variables it's made up of

        :type separator_vars: dict[string,None] or list[string]
        :return: a reference to the separator if it's found
        :rtype: Node
        """
        dict_sep = dict.fromkeys(separator_vars)
        found_sep = None
        for element in self._separators:
            if element.get_variables() == dict_sep:
                found_sep = element
                break

        if found_sep is None:
            raise AttributeError("Separator not valid")
        return found_sep

    def add_evidence(self, variable, value):
        """
        Add evidence to a variable in the JunctionTree by setting the table of the relative clique to the correct
        configuration(according to evidence)

        :type variable: string
        :param value: evidence to assert, can only be 0 or 1
        :type value: int
        :return: None
        """
        if variable not in self._variables:
            raise AttributeError("Variable not valid")
        if not (value == 0 or value == 1):
            raise AttributeError("Value not valid")

        # Find the clique to update
        chosen_clique = self._chosen_clique[variable]
        chosen_clique.received_evidence = True

        table = chosen_clique.get_prob_table()

        # Select which cell in the table has to be set to 0(those that contradict the evidence)
        coord_dict = {}
        for el in table.get_variables():
            if el != variable:
                coord_dict[el] = slice(None)
            else:
                coord_dict[el] = 1 - value

        table.set_probability_dict(coord_dict, 0)

        # Calculate P(U |e) = P(U,e)/P(e)
        table.divide_all(table.marginalize({variable: None}).get_prob(value))

    def get_joint_probability_table(self):
        """
        Returns the joint probability table of the whole Bayesian Net by multiplying all the tables of the cliques and
        dividing the result by all the tables of the separators

        :return: joint probability table
        :rtype: BeliefTable
        """
        # Multiply all the tables of the cliques
        result_table = BeliefTable(self._variables, np.ones((2,) * len(self._variables)))
        for clique in self._cliques:
            result_table = result_table.multiply_table(clique.get_prob_table())

        # Divide the result by all the separator tables
        for sep in self._separators:
            result_table = result_table.divide_table(sep.get_prob_table())

        return result_table

    def calculate_variables_probability_on_universe(self, variables):
        """
        Calculate the probabilities of one or more variables in the given configuration of the JunctionTree by
        calculating the joint prob table and then marginalizing on the given variables.
        Does not use the efficiency of Hugin propagation.

        :type variables: str or list[str] or dict[str,None]
        :return: probability table of the given variables
        :rtype: BeliefTable
        """
        if type(variables) is not str:
            variables = dict.fromkeys(variables)
        else:
            variables = {variables: None}

        if not variables.keys() <= self._variables.keys():
            raise AttributeError("Variables not valid", str(variables))

        table = self.get_joint_probability_table()

        return table.marginalize(variables)

    def calculate_variable_probability(self, variable):
        """
        Calculate the probabilities of the given variable by consulting the clique chosen for that variable during the
        JunctionTree creation. Needs Hugin propagation to give out correct results.

        :type variable: string
        :return: table over the single variable
        :rtype: BeliefTable
        """
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
        visited_labels = dict.fromkeys(self._cliques)
        for element in visited_labels:
            visited_labels[element] = False

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
        visited_labels = dict.fromkeys(self._cliques)
        for element in visited_labels:
            visited_labels[element] = False

        # Keep track of traversal to send back evidence to the correct node
        parents = dict.fromkeys(self._cliques)
        for element in parents:
            parents[element] = None

        # Perform a pseudo-BF traversal
        queue = []
        visited_labels[node] = True
        queue.append(node)

        while len(queue) != 0:
            v = queue.pop(0)

            # If a Node with evidence is found propagate its information over all its ancestors
            if v.received_evidence:
                ancestors = parents[v]
                while ancestors is not None:
                    common_separator = [x for x in v.get_neighbours() if x in ancestors.get_neighbours()][0]

                    JunctionTree.absorption(v, common_separator, ancestors)
                    ancestors = parents[ancestors]

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
        root = self._cliques[1]

        self.collect_evidence(root)
        self.distribute_evidence(root)

        # Normalize
        # Find normalizing constant by marginalizing on any variable
        variable_chosen = list(self._variables)[0]
        norm_table = self.calculate_variable_probability(variable_chosen)
        norm_constant = norm_table.get_prob(0) + norm_table.get_prob(1)

        for clique in self._cliques:
            clique.get_prob_table().divide_all(norm_constant)
        for separator in self._separators:
            separator.get_prob_table().divide_all(norm_constant)

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
        # Set all separators and cliques to 1
        for sep in self._separators:
            sep.get_prob_table().set_probability_coord((slice(None),)*len(sep.get_variables()), 1)

        for clique in self._cliques:
            clique.get_prob_table().set_probability_coord((slice(None),)*len(clique.get_variables()), 1)

        # Look up which node in the cluster tree was chosen for each variable to store its table in
        for variable in self._variables:
            clique = self._chosen_clique[variable]

            # Find the initial table in the bayes net
            table = bayes_net.get_table(variable)

            clique.set_prob_table(clique.get_prob_table().multiply_table(table))

        # Distribute initial information by allowing a round of message passing
        for node in self._cliques:
            node.received_evidence = True

        self.sum_propagate()

    def get_clique_from_dict(self, clique):
        """
        Get reference to a clique given the variables it's made up of

        :type clique: dict[string,None] or list[string]
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

        :type sep: dict[string,None] or list[string]
        :return: the separator or None if it doesn't exists
        :rtype: Node
        """
        sep = dict.fromkeys(sep)
        for element in self._separators:
            if element.get_variables() == sep:
                return element
        return None

    def __str__(self):
        """
        Represents the JunctionTree by listing all variables, all cliques, all separators and which clique was chosen
        for each variable
        """
        rstring = 'Variables:' + str(self._variables) + '\n'

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
            rstring += str(el) + ':' + self._chosen_clique[el].node_vars_to_string() + ', '
        rstring = rstring[:-2]

        return rstring


class Node(object):
    """
    Data structure that represents a clique or a separator in a JunctionTree
    """

    _table = None
    """
    BeliefTable of the Node, contains the variables that make up the clique/separator as well
    """

    _neighbours = []
    """
    List of other Nodes that represents the neighbours of the clique/separator
    """

    received_evidence = None
    """
    Boolean value that represents if the Node has received evidence, used in propagation
    """

    def __init__(self, table):
        """
        Initialize a Node to a given table,with no neighbours and with no evidence given

        :type table: BeliefTable
        """
        self._table = table
        self._neighbours = []
        self.received_evidence = False

    def get_variables(self):
        """
        Returns the variables that make up the Node

        :return: variables
        :rtype: dict[string, None]
        """
        return self._table.get_variables()

    def get_prob_table(self):
        """
        Returns the table relative to the Node

        :return: table
        :rtype: BeliefTable
        """
        return self._table

    def set_prob_table(self, table):
        """
        Sets the table relative to the Node

        :type table: BeliefTable
        :return: None
        """
        self._table = table

    def add_neighbour(self, node):
        """
        Add a neighbouring Node

        :type node: Node
        :return: None
        """
        self._neighbours.append(node)

    def get_neighbours(self):
        """
        Returns the list of neighbouring nodes

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
        return '.'.join(list(self.get_variables()))

