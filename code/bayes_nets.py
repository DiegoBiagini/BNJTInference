#
# This file contains the data structures used to represent and work on bayesian nets
#
from tables import BeliefTable

class BayesianNet(object):
    _graph = {}
    _tables = {}

    def __init__(self, graph=None):
        if graph is None:
            self._graph = {}
        else:
            self._graph = graph

    def add_variable(self, new_variable):
        if new_variable in self._graph.keys():
            raise AttributeError("Trying to add a variable that is already inside the net")
        else:
            self._graph[new_variable] = []
            self._tables[new_variable] = None

    def add_dependence(self, child, father):
        if father not in self._graph.keys():
            raise AttributeError("Invalid father")
        elif child not in self._graph.keys():
            raise AttributeError("Invalid child")
        else:
            self._graph[father].append(child)

    def add_prob_table(self, variable, table):
        if variable not in self._graph:
            raise AttributeError("Variable not valid")
        # Check if the passed table is correct
        fathers = self.get_fathers(variable)
        if fathers == []:
            vars_in_table = [variable]
        else:
            vars_in_table = fathers
            vars_in_table.insert(0, variable)

        if not(sorted(table.get_variables()) == sorted(vars_in_table)):
            raise AttributeError("Table for the variable is not valid")
        self._tables[variable] = table

    def get_table(self, variable):
        if variable not in self._tables.keys():
            raise AttributeError("Variable not valid")
        return self._tables[variable]

    def get_fathers(self, child):
        if child not in self._graph.keys():
            raise AttributeError("Child not found")

        fathers = []
        for el in self._graph:
            if child in self._graph[el]:
                fathers.append(el)

        return fathers

    def __str__(self):
        return str(self._graph)

    def print_u_probability(self):
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

        print(u_prob_string)

    def is_acyclic(self):
        """
        https://www.cs.hmc.edu/~keller/courses/cs60/s98/examples/acyclic/
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


class JunctionTree(object):
    _variables = {}

    _cliques = []
    _separators = []

    _chosen_clique = {}

    def __init__(self, variables):
        self._variables = variables

    def add_clique(self, clique):
        clique = dict.fromkeys(clique)
        if not clique.keys() <= self._variables.keys():
            raise AttributeError("The given clique is not valid for the junction tree")
        new_node = Node(BeliefTable(clique))
        self._cliques.append(new_node)

    def add_separator(self, separator):
        separator = dict.fromkeys(separator)
        if not separator.keys() <= self._variables.keys():
            raise AttributeError("The given separator is not valid for the junction tree")
        new_node = Node(BeliefTable(separator))
        self._separators.append(new_node)

    def add_link(self, clique, separator):
        true_clique = self.get_clique(clique)
        true_sep = self.get_separator(separator)
        self._add_link(true_clique, true_sep)

    def _add_link(self, clique, separator):
        if clique not in self._cliques or separator not in self._separators:
            raise AttributeError("Clique or separator not valid for the junction tree")
        if not(separator.get_variables().keys() <= clique.get_variables().keys()):
            raise AttributeError("Clique does not contain the variables in the separator")

        clique.add_neighbour(separator)
        separator.add_neighbour(clique)

    def set_variable_chosen_clique(self, variable, clique):
        true_clique = self.get_clique(clique)
        self._set_variable_chosen_clique(variable, true_clique)

    def _set_variable_chosen_clique(self, variable, clique):
        if variable not in self._variables.keys():
            raise AttributeError("Variable not valid")

        if clique not in self._cliques:
            raise AttributeError("Clique not valid")

        self._chosen_clique[variable] = clique

    def get_clique(self, clique):
        dict_clique = dict.fromkeys(clique)
        found_clique = None
        for element in self._cliques:
            if element.get_variables() == dict_clique:
                found_clique = element
                break

        if found_clique is None:
            raise AttributeError("Clique not valid")
        return found_clique

    def get_separator(self, separator):
        dict_sep = dict.fromkeys(separator)
        found_sep = None
        for element in self._separators:
            if element.get_variables() == dict_sep:
                found_sep = element
                break

        if found_sep is None:
            raise AttributeError("Separator not valid")
        return found_sep

    def add_evidence(self, variable, value):
        pass

    def propagate(self, first, separator, second):
        pass

    def __str__(self):
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
    _table = []
    _neighbours = []

    def __init__(self, table):
        self._table = table
        self._neighbours = []

    def get_variables(self):
        return self._table.get_variables()

    def get_prob_table(self):
        return self._table

    def add_neighbour(self, node):
        self._neighbours.append(node)

    def get_neighbours(self):
        return self._neighbours

    def node_vars_to_string(self):
        return ''.join(list(self.get_variables()))

