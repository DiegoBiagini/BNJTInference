import cProfile

import util as util
from bayes_nets import JunctionTree


def main():
    jtree : JunctionTree = None
    net, jtree = util.load_model("models/poker.dat")

    pr = cProfile.Profile()
    pr.enable()

    jtree.initialize_tables(net)

    jtree.add_evidence('MH', '2 alike')
    jtree.add_evidence('FC', '3 changed')
    jtree.add_evidence('SC', '1 changed')


    jtree.sum_propagate()

    print(jtree.calculate_variable_probability('BH'))



    pr.disable()
    # after your program ends
    pr.print_stats(sort="calls")





if __name__ == '__main__':
    main()
