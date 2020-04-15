import numpy as np

from tables import BeliefTable


def main():
    keywords = {'A', 'B'}
    table1 = np.arange(4).reshape(2,2)
    table2 = np.arange(4).reshape(2,2)

    b1 = BeliefTable(dict.fromkeys(keywords), table1)
    b2 = BeliefTable(dict.fromkeys(keywords), table2)

    print(b1.multiply(b2))


if __name__ == '__main__':
    main()
