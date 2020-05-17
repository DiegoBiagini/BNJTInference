#
# General utility functions
#

import pickle

import bayes_nets as bnet


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
