import unittest

import numpy as np

from bayes_nets import BayesianNet
from bayes_nets import JunctionTree
from tables import BeliefTable


class TableTests(unittest.TestCase):

    def test_creation(self):
        try:
            # Create valid single column belief table
            var = dict.fromkeys(['A'])
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

        res = b1.multiply_table(b2)
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

        res = b1.multiply_table(b2)
        self.assertEqual(res.get_prob((0, 0, 0)), 0)
        self.assertEqual(res.get_prob((0, 0, 1)), 0)
        self.assertEqual(res.get_prob((0, 1, 0)), 0)
        self.assertEqual(res.get_prob((0, 1, 1)), 1)
        self.assertEqual(res.get_prob((1, 0, 0)), 4)
        self.assertEqual(res.get_prob((1, 0, 1)), 6)
        self.assertEqual(res.get_prob((1, 1, 0)), 6)
        self.assertEqual(res.get_prob((1, 1, 1)), 9)

    def test_marginalization(self):
        # Check that it doesn't allow marginalization on sets greater than the variables of the table
        dict1 = dict.fromkeys(['A'])
        dict2 = dict.fromkeys(['A', 'B'])
        arr1 = np.arange(2).reshape(2)

        t1 = BeliefTable(dict1, arr1)
        self.assertRaises(AttributeError, t1.marginalize, dict2)

        # Test marginalization on single variable
        dict1 = dict.fromkeys(['A', 'B'])
        dict2 = dict.fromkeys(['A'])

        arr1 = np.arange(4).reshape((2, 2))

        t1 = BeliefTable(dict1, arr1)
        t2 = t1.marginalize(dict2)

        self.assertEqual(t2.get_prob(0), 1)
        self.assertEqual(t2.get_prob(1), 5)

        # Test marginalization on one variable from 3
        dict1 = dict.fromkeys(['A', 'B', 'C'])
        dict2 = dict.fromkeys(['C'])

        arr1 = np.arange(8).reshape((2, 2,2))

        t1 = BeliefTable(dict1, arr1)
        t2 = t1.marginalize(dict2)

        self.assertEqual(t2.get_prob(0), 12)
        self.assertEqual(t2.get_prob(1), 16)

        # Test marginalization on two variables
        dict1 = dict.fromkeys(['A', 'B', 'C'])
        dict2 = dict.fromkeys(['A', 'C'])

        arr1 = np.arange(8).reshape((2, 2,2))

        t1 = BeliefTable(dict1, arr1)
        t2 = t1.marginalize(dict2)

        self.assertEqual(t2.get_prob((0, 0)), 2)
        self.assertEqual(t2.get_prob((0, 1)), 4)
        self.assertEqual(t2.get_prob((1, 0)), 10)
        self.assertEqual(t2.get_prob((1, 1)), 12)

    def test_multiply_and_marginalize(self):
        # A reasonably big case I spent 30 minutes writing on paper, uses both multiplication and marginalization
        dict1 = dict.fromkeys(['D', 'S', 'L'])
        dict2 = dict.fromkeys(['S', 'H'])
        dict3 = dict.fromkeys(['D'])
        dict4 = dict.fromkeys(['S'])

        holes = dict.fromkeys(['H'])

        arr1 = np.array([[[0.02, 0.98], [0.9, 0.1]], [[0.85, 0.15], [0.95, 0.05]]])
        arr2 = np.array([[0.8, 0.2], [0.3, 0.7]])
        arr3 = np.array([0.9, 0.1])
        arr4 = np.array([0.9, 0.1])

        t1 = BeliefTable(dict1, arr1)
        t2 = BeliefTable(dict2, arr2)
        t3 = BeliefTable(dict3, arr3)
        t4 = BeliefTable(dict4, arr4)

        t5 = t1.multiply_table(t2).multiply_table(t3).multiply_table(t4)
        t5 = t5.marginalize(holes)
        self.assertAlmostEqual(t5.get_prob(0), 0.75)
        self.assertAlmostEqual(t5.get_prob(1), 0.25)


class BayesianNetTests(unittest.TestCase):

    def test_creation(self):
        # Test creating empty net and adding variables
        try:
            net = BayesianNet()
            net.add_variable('A')
            net.add_variable('B')
        except AttributeError:
            self.fail("Error creating empty BayesianNet and adding variables")

        # Test error raising when inserting the same variable twice
        net = BayesianNet()
        net.add_variable('A')
        self.assertRaises(AttributeError, net.add_variable, 'A')

    def test_cyclic(self):
        # Test on acyclic graph
        net = BayesianNet()
        net.add_variable('A')
        net.add_variable('B')
        net.add_variable('C')
        net.add_dependence('B', 'A')
        net.add_dependence('C', 'B')

        self.assertTrue(net.is_acyclic())

        # Test on cyclic graph
        net = BayesianNet()
        net.add_variable('A')
        net.add_variable('B')
        net.add_variable('C')
        net.add_dependence('B', 'A')
        net.add_dependence('C', 'B')
        net.add_dependence('A', 'C')

        self.assertFalse(net.is_acyclic())

    def test_table_assignment(self):
        # Check if assigning tables via identifiers works correctly even when the order of variables is wrong

        tS = BeliefTable(dict.fromkeys('S'), np.zeros(2))
        tD = BeliefTable(dict.fromkeys('D'), np.zeros(2))
        tL = BeliefTable(dict.fromkeys(['S', 'D', 'L']), np.zeros((2, 2, 2)))

        tS.set_probability_dict({'S': 0}, 0.9)
        tS.set_probability_dict({'S': 1}, 0.1)

        tD.set_probability_dict({'D': 0}, 0.9)
        tD.set_probability_dict({'D': 1}, 0.1)

        tL.set_probability_dict({'S': 0, 'D': 0, 'L': 0}, 0.98)
        tL.set_probability_dict({'S': 0, 'D': 0, 'L': 1}, 0.02)
        tL.set_probability_dict({'S': 0, 'D': 1, 'L': 0}, 0.15)
        tL.set_probability_dict({'D': 1, 'L': 1, 'S': 0}, 0.85)
        tL.set_probability_dict({'L': 0, 'S': 1, 'D': 0}, 0.1)
        tL.set_probability_dict({'S': 1, 'D': 0, 'L': 1}, 0.9)
        tL.set_probability_dict({'S': 1, 'D': 1, 'L': 0}, 0.05)
        tL.set_probability_dict({'S': 1, 'D': 1, 'L': 1}, 0.95)

        net = BayesianNet()
        net.add_variable('L')
        net.add_variable('S')
        net.add_variable('D')

        net.add_dependence('L', 'S')
        net.add_dependence('L', 'D')

        net.add_prob_table('L', tL)
        net.add_prob_table('S', tS)
        net.add_prob_table('D', tD)

        self.assertEqual(str(tS), str(net.get_table('S')))
        self.assertEqual(str(tD), str(net.get_table('D')))
        self.assertEqual(str(tL), str(net.get_table('L')))


class JunctionTreeTest(unittest.TestCase):

    def test_creation(self):
        # Just check that creation raises no errors
        variables = dict.fromkeys(['S', 'D', 'H', 'L'])
        jtree = JunctionTree(variables)

        jtree.add_clique(dict.fromkeys(['S', 'H']))
        jtree.add_clique(dict.fromkeys(['S', 'D', 'L']))
        jtree.add_separator(dict.fromkeys(['S']))

        jtree.add_link(['S', 'H'], ['S'])
        jtree.add_link(['S', 'D', 'L'], ['S'])

        jtree.set_variable_chosen_clique('H', ['S', 'H'])
        jtree.set_variable_chosen_clique('S', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('D', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('L', ['S', 'D', 'L'])







if __name__ == '__main__':
    unittest.main()



