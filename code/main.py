from bayes_nets import BayesianNet


def main():
    net = BayesianNet()
    net.add_variable('A')
    net.add_variable('B')
    net.add_variable('C')
    net.add_dependence('B', 'A')
    net.add_dependence('C', 'B')
    net.add_dependence('A','C')

    print(net.is_acyclic())


if __name__ == '__main__':
    main()
