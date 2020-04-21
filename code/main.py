import numpy as np

from bayes_nets import BayesianNet
from bayes_nets import JunctionTree
from tables import BeliefTable


def main():
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
    tL.set_probability_dict({'D': 1, 'L': 1, 'S': 0}, 0.85)
    tL.set_probability_dict({'L': 0, 'S': 1, 'D': 0}, 0.1)
    tL.set_probability_dict({'S': 1, 'D': 0, 'L': 1}, 0.9)
    tL.set_probability_dict({'S': 1, 'D': 1, 'L': 0}, 0.05)
    tL.set_probability_dict({'S': 1, 'D': 1, 'L': 1}, 0.95)

    net = BayesianNet()
    net.add_variable('L')
    net.add_variable('S')
    net.add_variable('D')

    net.add_dependence('L', 'S')
    net.add_dependence('L', 'D')

    net.add_prob_table('L', tL)
    net.add_prob_table('S', tS)
    net.add_prob_table('D', tD)

    jtree = JunctionTree(dict.fromkeys(['S', 'D', 'L']))
    jtree.add_clique(['S', 'D', 'L'])
    jtree.set_variable_chosen_clique('S', ['S', 'D', 'L'])
    jtree.set_variable_chosen_clique('D', ['S', 'D', 'L'])
    jtree.set_variable_chosen_clique('L', ['S', 'D', 'L'])

    jtree.initialize_tables(net)
    jtree.add_evidence('L', 0)

    Stable = jtree.calculate_variables_probability(['S'])
    DTable = jtree.calculate_variables_probability(['D'])
    print(Stable)
    print(DTable)


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


    # Insert evidence that L=1
    prodTable = tS.multiply_table(tD).multiply_table(tL)
    nT = tS.multiply_table(tD).multiply_table(tL)

    nT.set_probability_dict({'S':slice(None), 'D':slice(None), 'L':0},0)

    nT.divide_all(prodTable.marginalize(dict.fromkeys(['L'])).get_prob(1))
    print(nT.marginalize(dict.fromkeys(['D'])))
    print(nT.marginalize(dict.fromkeys(['S'])))
    print(nT.marginalize(dict.fromkeys(['L'])))


if __name__ == '__main__':
    main()
