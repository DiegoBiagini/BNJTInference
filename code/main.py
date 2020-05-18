import cProfile

import models
import util as util
from bayes_nets import BayesianNet
from bayes_nets import JunctionTree
from tables import BeliefTable
from tables import Variable


def main():

    net, jtree = models.build_fire()
    util.serialize_model(net, jtree, "models/fire.dat")

    pr = cProfile.Profile()
    pr.enable()

    jtree.initialize_tables(net)
    jtree.add_evidence('R', 'true')
    #jtree.add_evidence('S', 'true')

    jtree.sum_propagate()

    print(jtree.calculate_variable_probability('A'))
    print(jtree.calculate_variable_probability('F'))
    print(jtree.calculate_variable_probability('L'))
    print(jtree.calculate_variable_probability('R'))
    print(jtree.calculate_variable_probability('S'))
    print(jtree.calculate_variable_probability('T'))

    pr.disable()
    # after your program ends
    pr.print_stats(sort="calls")








    """
    util.serialize_model(model[0], model[1], "models/cancer.dat")

    net, jtree = util.load_model("models/cancer.dat")

    jtree.initialize_tables(net)

    jtree.add_evidence('C', 'Present')
    jtree.add_evidence('H', 'Present')
    jtree.add_evidence('MC', 'Absent')
    jtree.sum_propagate()



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
    
    """



def build_random_3():
    A = Variable('A', ['a0', 'a1'])
    B = Variable('B', ['b0', 'b1', 'b2'])
    C = Variable('C', ['c0', 'c1', 'c2'])


    tA = BeliefTable([A])
    tA.set_probability_dict({'A':'a0'},0.3)
    tA.set_probability_dict({'A':'a1'},0.7)

    tB = BeliefTable([B])
    tB.set_probability_dict({'B':'b0'},0.1)
    tB.set_probability_dict({'B':'b1'},0.4)
    tB.set_probability_dict({'B':'b2'},0.5)

    tC = BeliefTable([A,B,C])
    tC.set_probability_dict({'A':'a0','B':'b0', 'C':'c0'}, 0.1)
    tC.set_probability_dict({'A':'a0','B':'b0', 'C':'c1'}, 0.2)
    tC.set_probability_dict({'A':'a0','B':'b0', 'C':'c2'}, 0.7)
    tC.set_probability_dict({'A':'a0','B':'b1', 'C':'c0'}, 0.3)
    tC.set_probability_dict({'A':'a0','B':'b1', 'C':'c1'}, 0.4)
    tC.set_probability_dict({'A':'a0','B':'b1', 'C':'c2'}, 0.3)
    tC.set_probability_dict({'A':'a0','B':'b2', 'C':'c0'}, 0.5)
    tC.set_probability_dict({'A':'a0','B':'b2', 'C':'c1'}, 0.4)
    tC.set_probability_dict({'A':'a0','B':'b2', 'C':'c2'}, 0.1)
    tC.set_probability_dict({'A':'a1','B':'b0', 'C':'c0'}, 0.2)
    tC.set_probability_dict({'A':'a1','B':'b0', 'C':'c1'}, 0.3)
    tC.set_probability_dict({'A':'a1','B':'b0', 'C':'c2'}, 0.5)
    tC.set_probability_dict({'A':'a1','B':'b1', 'C':'c0'}, 0.4)
    tC.set_probability_dict({'A':'a1','B':'b1', 'C':'c1'}, 0.5)
    tC.set_probability_dict({'A':'a1','B':'b1', 'C':'c2'}, 0.1)
    tC.set_probability_dict({'A':'a1','B':'b2', 'C':'c0'}, 0.6)
    tC.set_probability_dict({'A':'a1','B':'b2', 'C':'c1'}, 0.3)
    tC.set_probability_dict({'A':'a1','B':'b2', 'C':'c2'}, 0.1)

    net = BayesianNet()
    net.add_variable(A)
    net.add_variable(B)
    net.add_variable(C)

    net.add_dependence(C,A)
    net.add_dependence(C,B)

    net.add_prob_table(A,tA)
    net.add_prob_table(B,tB)
    net.add_prob_table(C,tC)

    jtree = JunctionTree([A,B,C])
    jtree.add_clique([A,B,C])

    jtree.set_variable_chosen_clique(A,[A,B,C])
    jtree.set_variable_chosen_clique(B,[A,B,C])
    jtree.set_variable_chosen_clique(C,[A,B,C])




if __name__ == '__main__':
    main()
