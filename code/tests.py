import unittest

import numpy as np

from tables import BeliefTable


class TableTests(unittest.TestCase):

    def test_creation(self):
        try:
            # Create valid single column belief table
            var = dict.fromkeys({'A'})
            table = np.ones(2)
            btable = BeliefTable(var, table)

            # Create valid 3 variable array
            var = dict.fromkeys({'A', 'B', 'C'})
            table = np.ones((2, 2, 2))
            btable = BeliefTable(var, table)
        except AttributeError:
            self.fail("Error during table creation!")


if __name__ == '__main__':
    unittest.main()



