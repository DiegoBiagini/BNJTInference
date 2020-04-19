import numpy as np

from bayes_nets import BayesianNet
from tables import BeliefTable


def main():
    pass


def sample_case():
    tS = BeliefTable(dict.fromkeys('S'), np.zeros(2))
    tD = BeliefTable(dict.fromkeys('D'), np.zeros(2))
    tL = BeliefTable(dict.fromkeys(['S', 'D', 'L']), np.zeros((2, 2, 2)))

    tS.set_probability_dict({'S': 0}, 0.9)
    tS.set_probability_dict({'S': 1}, 0.1)

    tD.set_probability_dict({'D': 0}, 0.9)
    tD.set_probability_dict({'D': 1}, 0.1)

    tL.set_probability_dict({'S': 0, 'D': 0, 'L': 0}, 0.98)
    tL.set_probability_dict({'S': 0, 'D': 0, 'L': 1}, 0.02)
    tL.set_probability_dict({'S': 0, 'D': 1, 'L': 0}, 0.15)
    tL.set_probability_dict({'S': 0, 'D': 1, 'L': 1}, 0.85)
    tL.set_probability_dict({'S': 1, 'D': 0, 'L': 0}, 0.1)
    tL.set_probability_dict({'S': 1, 'D': 0, 'L': 1}, 0.9)
    tL.set_probability_dict({'S': 1, 'D': 1, 'L': 0}, 0.05)
    tL.set_probability_dict({'S': 1, 'D': 1, 'L': 1}, 0.95)

    net = BayesianNet()
    net.add_variable('S')
    net.add_variable('D')
    net.add_variable('L')

    net.add_dependence('L', 'S')
    net.add_dependence('L', 'D')

    net.add_prob_table('S', tS)
    net.add_prob_table('D', tD)
    net.add_prob_table('L', tL)

if __name__ == '__main__':
    main()
