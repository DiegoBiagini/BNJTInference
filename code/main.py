import util as util
from bayes_nets import BayesianNet
from bayes_nets import JunctionTree
from tables import BeliefTable


def main():
    model = build_cancer()

    util.serialize_model(model[0], model[1], "models/cancer.dat")

    net, jtree = util.load_model("models/cancer.dat")
    jtree.initialize_tables(net)

    MCTable = jtree.calculate_variable_probability('MC')
    STable = jtree.calculate_variable_probability('T')
    TTable = jtree.calculate_variable_probability('S')
    CTable = jtree.calculate_variable_probability('C')
    HTable = jtree.calculate_variable_probability('H')

    print(MCTable)
    print(STable)
    print(TTable)
    print(CTable)
    print(HTable)



def build_cancer():
    tMC = BeliefTable(['MC'])
    tS = BeliefTable(['S', 'MC'])
    tT = BeliefTable(['T', 'MC'])
    tC = BeliefTable(['C', 'S', 'T'])
    tH = BeliefTable(['H', 'T'])

    tMC.set_probability_dict({'MC': 0}, 0.8)
    tMC.set_probability_dict({'MC': 1}, 0.2)

    tS.set_probability_dict({'S': 0, 'MC': 0}, 0.8)
    tS.set_probability_dict({'S': 0, 'MC': 1}, 0.2)
    tS.set_probability_dict({'S': 1, 'MC': 0}, 0.2)
    tS.set_probability_dict({'S': 1, 'MC': 1}, 0.8)

    tT.set_probability_dict({'T': 0, 'MC': 0}, 0.95)
    tT.set_probability_dict({'T': 0, 'MC': 1}, 0.8)
    tT.set_probability_dict({'T': 1, 'MC': 0}, 0.05)
    tT.set_probability_dict({'T': 1, 'MC': 1}, 0.2)

    tC.set_probability_dict({'C': 0, 'S': 0, 'T': 0}, 0.95)
    tC.set_probability_dict({'C': 0, 'S': 0, 'T': 1}, 0.2)
    tC.set_probability_dict({'C': 0, 'S': 1, 'T': 0}, 0.2)
    tC.set_probability_dict({'C': 0, 'S': 1, 'T': 1}, 0.2)
    tC.set_probability_dict({'C': 1, 'S': 0, 'T': 0}, 0.05)
    tC.set_probability_dict({'C': 1, 'S': 0, 'T': 1}, 0.8)
    tC.set_probability_dict({'C': 1, 'S': 1, 'T': 0}, 0.8)
    tC.set_probability_dict({'C': 1, 'S': 1, 'T': 1}, 0.8)

    tH.set_probability_dict({'H': 0, 'T': 0}, 0.4)
    tH.set_probability_dict({'H': 0, 'T': 1}, 0.2)
    tH.set_probability_dict({'H': 1, 'T': 0}, 0.6)
    tH.set_probability_dict({'H': 1, 'T': 1}, 0.8)

    net = BayesianNet()
    net.add_variable('MC')
    net.add_variable('S')
    net.add_variable('T')
    net.add_variable('C')
    net.add_variable('H')

    net.add_dependence('S', 'MC')
    net.add_dependence('T', 'MC')
    net.add_dependence('C', 'S')
    net.add_dependence('C', 'T')
    net.add_dependence('H', 'T')

    net.add_prob_table('MC', tMC)
    net.add_prob_table('S', tS)
    net.add_prob_table('T', tT)
    net.add_prob_table('C', tC)
    net.add_prob_table('H', tH)
    # print(net.get_U_probability_string())

    jtree = JunctionTree(['MC', 'S', 'T', 'C', 'H'])

    jtree.add_clique(['T', 'S', 'C'])
    jtree.add_clique(['T', 'S', 'MC'])
    jtree.add_clique(['T', 'H'])

    jtree.add_separator(['T', 'S'])
    jtree.add_separator(['T'])

    jtree.add_link(['T', 'S', 'MC'], ['T', 'S'])
    jtree.add_link(['T', 'S', 'C'], ['T', 'S'])
    jtree.add_link(['T', 'S', 'MC'], ['T'])
    jtree.add_link(['T', 'H'], ['T'])

    jtree.set_variable_chosen_clique('MC', ['T', 'S', 'MC'])
    jtree.set_variable_chosen_clique('S', ['T', 'S', 'MC'])
    jtree.set_variable_chosen_clique('T', ['T', 'S', 'MC'])
    jtree.set_variable_chosen_clique('C', ['T', 'S', 'C'])
    jtree.set_variable_chosen_clique('H', ['T', 'H'])

    return net, jtree



if __name__ == '__main__':
    main()
