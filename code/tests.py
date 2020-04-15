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

    def test_multiplication(self):
        # Multiplication with same variables
        keywords = ['A', 'B']
        table1 = np.arange(4).reshape(2, 2)
        table2 = np.arange(4).reshape(2, 2)

        b1 = BeliefTable(dict.fromkeys(keywords), table1)
        b2 = BeliefTable(dict.fromkeys(keywords), table2)

        res = b1.multiply(b2)
        self.assertEqual(res.get_prob((0, 0)), 0)
        self.assertEqual(res.get_prob((0, 1)), 1)
        self.assertEqual(res.get_prob((1, 0)), 4)
        self.assertEqual(res.get_prob((1, 1)), 9)

        # Multiplication with different variables
        keywords1 = ['A', 'B']
        keywords2 = ['A', 'C']
        table1 = np.arange(4).reshape(2, 2)
        table2 = np.arange(4).reshape(2, 2)

        b1 = BeliefTable(dict.fromkeys(keywords1), table1)
        b2 = BeliefTable(dict.fromkeys(keywords2), table2)

        res = b1.multiply(b2)
        self.assertEqual(res.get_prob((0, 0, 0)), 0)
        self.assertEqual(res.get_prob((0, 0, 1)), 0)
        self.assertEqual(res.get_prob((0, 1, 0)), 0)
        self.assertEqual(res.get_prob((0, 1, 1)), 1)
        self.assertEqual(res.get_prob((1, 0, 0)), 4)
        self.assertEqual(res.get_prob((1, 0, 1)), 6)
        self.assertEqual(res.get_prob((1, 1, 0)), 6)
        self.assertEqual(res.get_prob((1, 1, 1)), 9)



if __name__ == '__main__':
    unittest.main()



