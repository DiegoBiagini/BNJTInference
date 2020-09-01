#
# General utility functions
#

import pickle

import numpy as np

import bayes_nets as bnet
import tables as tables


def subtract_ordered_dict(d1, d2):
    """
    Executes set subtraction between two dictionaries used as ordered sets

    :type d1: dict
    :type d2: dict
    :return: the result of set subtraction
    :rtype: dict
    """

    return dict.fromkeys([el for el in d1.keys() if el not in d2.keys()])


def serialize_model(bayes_net, junction_tree, filename):
    """
    Save a bayesian net and its junction tree into a file

    :type bayes_net: bnet.BayesianNet
    :type junction_tree:  bnet.JunctionTree
    :type filename: string
    :return: None
    """

    data = [bayes_net, junction_tree]
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_model(filename):
    """
    Load a bayesian net and its junction tree from file, returns a tuple (BayesianNet, JunctionTree)

    :type filename: string
    :rtype: tuple(bnet.BayesianNet, bnet.JunctionTree)
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def shape_to_list_of_entries(shape):
    """
    Returns a list containing all coordinates of a np table of the given shape
    e.g. shape = (2,3,2) the result is [(0,0,0), (0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),(1,0,0) ... (1,2,1)]
    This is kinda like successive divisions with different bases each time

    :type shape: tuple(int)
    :rtype: list[tuple(int)]
    """

    max_number = np.prod(shape)

    # Pre calculate the term you have to divide each time
    cache_divider = {}
    for i in range(len(shape)):
        divider = 1
        for j in range(i + 1, len(shape)):
            divider *= shape[j]
        cache_divider[i] = divider

    entry_list = []
    for i in range(max_number):
        new_entry = []

        dividend = i
        for j in range(len(shape)):

            divider = cache_divider[j]
            new_entry.append(dividend//divider)

            dividend = dividend % divider

        entry_list.append(tuple(new_entry))

    return entry_list


def get_shape_from_var_dict(variables):
    """
    Returns the shape of a supposed table that uses the given variables
    e.g. if tAB uses A and B and each can take 3 values the final shape will be (3,3)

    :type variables: dict[tables.Variable,None] or lis[tables.Variable]
    :return: shape
    :rtype: tuple[int]
    """
    variables = dict.fromkeys(variables)
    shape = []
    for el in variables.keys():
        shape.append(el.get_cardinality())

    return tuple(shape)


def get_size_from_var_dict(variables):
    """
    Returns the size of a supposed table that uses the given variables
    e.g. if tAB uses A and B and each can take 3 values the final size will be 9

    :type variables: dict[tables.Variable,None] or lis[tables.Variable]
    :return: size
    :rtype: int
    """
    return np.prod(get_shape_from_var_dict(variables))