#
# This file contains the data structures used to represent and work on bayesian nets
#


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

        # If the graph has no nodes is acyclic
        return True



class JunctionTree(object):

    _cliques = []
    _separators = []

    def __init__(self):
        print("Init")


class Node(object):
    _variables = {}
    _table = []

    def __init__(self, variables, table):
        self._variables = variables
        self._table = table



