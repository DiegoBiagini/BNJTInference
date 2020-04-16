import numpy as np

from tables import BeliefTable


def main():
    # Test marginalization on two variables
    dict1 = dict.fromkeys(['A', 'B', 'C'])
    dict2 = dict.fromkeys(['A', 'C'])

    arr1 = np.arange(8).reshape((2, 2, 2))

    t1 = BeliefTable(dict1, arr1)
    t2 = t1.marginalize(dict2)
    print(t1)
    print(t2)


if __name__ == '__main__':
    main()
