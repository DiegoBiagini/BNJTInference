#
# This file contains the junction tree data structure, the definition of its nodes and most of the functions applicable
# to them
#


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



