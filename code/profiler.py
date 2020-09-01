import cProfile

import util as util


# File used to check inference speed

def main():

    # Load the model you want to profile
    net, jtree = util.load_model("models/poker.dat")

    # Without Hugin Prop
    pr = cProfile.Profile()
    jtree.initialize_tables(net)

    pr.enable()

    jtree.add_evidence('FC', '1 changed')
    jtree.add_evidence('SC', '0 changed')
    jtree.add_evidence('MH', 'flush')
    result = jtree.calculate_variable_probability_on_universe('BH')

    pr.disable()
    pr.print_stats(sort="calls")
    print(result)

    # With Hugin Prop
    pr = cProfile.Profile()

    jtree.initialize_tables(net)

    pr.enable()

    jtree.add_evidence('FC', '1 changed')
    jtree.add_evidence('SC', '0 changed')
    jtree.add_evidence('MH', 'flush')

    jtree.sum_propagate()
    result = jtree.calculate_variable_probability('BH')

    pr.disable()
    pr.print_stats(sort="calls")
    print(result)



if __name__ == '__main__':
    main()
