import warnings

import menu


def main():
    warnings.filterwarnings('error')
    menu.show_first_menu()
    '''
    net, jtree = models.build_studfarm()
    util.serialize_model(net, jtree, "models/studfarm.dat")
    net, jtree = util.load_model("models/studfarm.dat")


    pr = cProfile.Profile()
    pr.enable()
    #print(net)
    jtree.initialize_tables(net)
    #print(jtree)

    jtree.add_evidence('J', 'sick')
    jtree.add_evidence('A', 'carrier')
    jtree.add_evidence('E', 'pure')

    jtree.sum_propagate()

    print(jtree.calculate_variable_probability('A'))
    print(jtree.calculate_variable_probability('B'))
    print(jtree.calculate_variable_probability('C'))
    print(jtree.calculate_variable_probability('D'))
    print(jtree.calculate_variable_probability('E'))
    print(jtree.calculate_variable_probability('F'))
    print(jtree.calculate_variable_probability('G'))
    print(jtree.calculate_variable_probability('H'))
    print(jtree.calculate_variable_probability('I'))
    print(jtree.calculate_variable_probability('J'))
    print(jtree.calculate_variable_probability('K'))
    print(jtree.calculate_variable_probability('L'))

    pr.disable()
    # after your program ends
    pr.print_stats(sort="calls")
    '''


if __name__ == '__main__':
    main()
