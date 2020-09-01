import unittest

import numpy as np

from bayes_nets import BayesianNet
from bayes_nets import JunctionTree
from tables import BeliefTable
from tables import Variable


class TableTests(unittest.TestCase):

    def setUp(self):
        # 3 test variables
        self.A = Variable('A', 'A', [0, 1])
        self.B = Variable('B', 'B', [0, 1])
        self.C = Variable('C', 'C', [0, 1])

    def test_creation(self):
        try:
            # Create valid single column belief table
            table = np.ones(2)
            btable = BeliefTable([self.A], table)

            # Create valid 3 variable array

            table = np.ones((2, 2, 2))
            btable = BeliefTable([self.A, self.B, self.C], table)
        except AttributeError:
            self.fail("Error during table creation!")

    def test_multiplication(self):

        # Multiplication with same variables
        keywords = [self.A, self.B]
        table1 = np.arange(4).reshape(2, 2)
        table2 = np.arange(4).reshape(2, 2)

        b1 = BeliefTable(keywords, table1)
        b2 = BeliefTable(keywords, table2)

        res = b1.multiply_table(b2)
        self.assertEqual(res.get_prob((0, 0)), 0)
        self.assertEqual(res.get_prob((0, 1)), 1)
        self.assertEqual(res.get_prob((1, 0)), 4)
        self.assertEqual(res.get_prob((1, 1)), 9)

        # Multiplication with different variables
        keywords1 = [self.A, self.B]
        keywords2 = [self.A, self.C]
        table1 = np.arange(4).reshape(2, 2)
        table2 = np.arange(4).reshape(2, 2)

        b1 = BeliefTable(keywords1, table1)
        b2 = BeliefTable(keywords2, table2)

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
        dict1 = dict.fromkeys([self.A])
        dict2 = dict.fromkeys([self.A, self.B])
        arr1 = np.arange(2).reshape(2)

        t1 = BeliefTable(dict1, arr1)
        self.assertRaises(AttributeError, t1.marginalize, dict2)

        # Test marginalization on single variable
        dict1 = dict.fromkeys([self.A, self.B])
        dict2 = dict.fromkeys([self.A])

        arr1 = np.arange(4).reshape((2, 2))

        t1 = BeliefTable(dict1, arr1)
        t2 = t1.marginalize(dict2)

        self.assertEqual(t2.get_prob(0), 1)
        self.assertEqual(t2.get_prob(1), 5)

        # Test marginalization on one variable from 3
        dict1 = dict.fromkeys([self.A, self.B, self.C])
        dict2 = dict.fromkeys([self.C])

        arr1 = np.arange(8).reshape((2, 2,2))

        t1 = BeliefTable(dict1, arr1)
        t2 = t1.marginalize(dict2)

        self.assertEqual(t2.get_prob(0), 12)
        self.assertEqual(t2.get_prob(1), 16)

        # Test marginalization on two variables
        dict1 = dict.fromkeys([self.A, self.B, self.C])
        dict2 = dict.fromkeys([self.A, self.C])

        arr1 = np.arange(8).reshape((2, 2,2))

        t1 = BeliefTable(dict1, arr1)
        t2 = t1.marginalize(dict2)

        self.assertEqual(t2.get_prob((0, 0)), 2)
        self.assertEqual(t2.get_prob((0, 1)), 4)
        self.assertEqual(t2.get_prob((1, 0)), 10)
        self.assertEqual(t2.get_prob((1, 1)), 12)

    def test_multiply_and_marginalize(self):
        # A reasonably big case I spent 30 minutes writing on paper, uses both multiplication and marginalization
        D = Variable('D', 'D', [0, 1])
        S = Variable('S', 'S', [0, 1])
        L = Variable('L', 'L', [0, 1])
        H = Variable('H', 'H', [0, 1])

        dict1 = dict.fromkeys([D, S, L])
        dict2 = dict.fromkeys([S, H])
        dict3 = dict.fromkeys([D])
        dict4 = dict.fromkeys([S])

        holes = dict.fromkeys([H])

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

    def setUp(self):
        self.A = Variable('A', 'A', [0, 1])
        self.B = Variable('B', 'B', [0, 1])
        self.C = Variable('C', 'C', [0, 1])

    def test_creation(self):
        # Test creating empty net and adding variables
        try:
            net = BayesianNet()
            net.add_variable(self.A)
            net.add_variable(self.B)
        except AttributeError:
            self.fail("Error creating empty BayesianNet and adding variables")

        # Test error raising when inserting the same variable twice
        net = BayesianNet()
        net.add_variable(self.A)
        self.assertRaises(AttributeError, net.add_variable, self.A)

    def test_table_assignment(self):
        # Check if assigning tables via identifiers works correctly even when the order of variables is wrong
        S = Variable('S', 'S', [0, 1])
        D = Variable('D', 'D', [0, 1])
        L = Variable('L', 'L', [0, 1])

        tS = BeliefTable(dict.fromkeys([S]), np.zeros(2))
        tD = BeliefTable(dict.fromkeys([D]), np.zeros(2))
        tL = BeliefTable(dict.fromkeys([S, D, L]), np.zeros((2, 2, 2)))

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
        net.add_variable(L)
        net.add_variable(S)
        net.add_variable(D)

        net.add_dependence('L', 'S')
        net.add_dependence('L', 'D')

        net.add_prob_table('L', tL)
        net.add_prob_table('S', tS)
        net.add_prob_table('D', tD)

        self.assertEqual(str(tS), str(net.get_table(S)))
        self.assertEqual(str(tD), str(net.get_table(D)))
        self.assertEqual(str(tL), str(net.get_table(L)))


class JunctionTreeTest(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.__init__(self)
        self.S = Variable('S', 'S', [0, 1])
        self.D = Variable('D', 'D', [0, 1])
        self.H = Variable('H', 'H', [0, 1])
        self.L = Variable('L', 'L', [0, 1])

    def test_creation(self):
        # Just check that creation raises no errors
        variables = dict.fromkeys([self.S, self.D, self.H, self.L])
        jtree = JunctionTree(variables)

        # Might as well test all 3 modes of adding var specifications
        jtree.add_clique(dict.fromkeys([self.S, self.H]))
        jtree.add_clique(['S', 'D', 'L'])
        jtree.add_separator([self.S])

        jtree.add_link(['S', 'H'], ['S'])
        jtree.add_link(['S', 'D', 'L'], ['S'])

        jtree.set_variable_chosen_clique('H', ['S', 'H'])
        jtree.set_variable_chosen_clique('S', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('D', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('L', ['S', 'D', 'L'])

    def test_bnet_linking(self):
        # Test if it's possible to calculate the probability of a variable given the bayesian net
        # No message passing, single cluster

        tS = BeliefTable([self.S], np.zeros(2))
        tD = BeliefTable([self.D], np.zeros(2))
        tL = BeliefTable(dict.fromkeys([self.S, self.D, self.L]), np.zeros((2, 2, 2)))

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
        net.add_variable(self.L)
        net.add_variable(self.S)
        net.add_variable(self.D)

        net.add_dependence('L', 'S')
        net.add_dependence('L', 'D')

        net.add_prob_table('L', tL)
        net.add_prob_table('S', tS)
        net.add_prob_table('D', tD)

        jtree = JunctionTree(dict.fromkeys([self.S, self.D, self.L]))
        jtree.add_clique(['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('S', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('D', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('L', ['S', 'D', 'L'])

        jtree.initialize_tables(net)

        Ltable = jtree.calculate_variable_probability('L')
        self.assertAlmostEqual(Ltable.get_prob(0), 0.8168)
        self.assertAlmostEqual(Ltable.get_prob(1), 0.1832)

    def test_inserting_evidence(self):
        # Cluster tree with only one node
        tS = BeliefTable([self.S], np.zeros(2))
        tD = BeliefTable([self.D], np.zeros(2))
        tL = BeliefTable(dict.fromkeys([self.S, self.D, self.L]), np.zeros((2, 2, 2)))

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
        net.add_variable(self.L)
        net.add_variable(self.S)
        net.add_variable(self.D)

        net.add_dependence('L', 'S')
        net.add_dependence('L', 'D')

        net.add_prob_table('L', tL)
        net.add_prob_table('S', tS)
        net.add_prob_table('D', tD)

        jtree = JunctionTree(dict.fromkeys([self.S, self.D, self.L]))
        jtree.add_clique(['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('S', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('D', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('L', ['S', 'D', 'L'])

        jtree.initialize_tables(net)

        # Add evidence to one node
        jtree.add_evidence('L', 0)
        Stable = jtree.calculate_variable_probability('S')
        Dtable = jtree.calculate_variable_probability('D')

        self.assertAlmostEqual(Stable.get_prob(0), 0.9884, 4)
        self.assertAlmostEqual(Stable.get_prob(1), 0.0116, 4)

        self.assertAlmostEqual(Dtable.get_prob(0), 0.9829, 4)
        self.assertAlmostEqual(Dtable.get_prob(1), 0.0171, 4)

        # Add evidence to another one
        jtree.add_evidence('S', 0)

        Stable = jtree.calculate_variable_probability('S')
        Dtable = jtree.calculate_variable_probability('D')

        self.assertAlmostEqual(Stable.get_prob(0), 1, 4)
        self.assertAlmostEqual(Stable.get_prob(1), 0, 4)

        self.assertAlmostEqual(Dtable.get_prob(0), 0.9833, 4)
        self.assertAlmostEqual(Dtable.get_prob(1), 0.0167, 4)

    def test_collect_and_distribute(self):
        # Apple tree example, 2 clusters
        tS = BeliefTable(dict.fromkeys([self.S]), np.zeros(2))
        tD = BeliefTable(dict.fromkeys([self.D]), np.zeros(2))
        tL = BeliefTable(dict.fromkeys([self.S, self.D, self.L]), np.zeros((2, 2, 2)))
        tH = BeliefTable(dict.fromkeys([self.H, self.S]))

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

        tH.set_probability_dict({'H': 0, 'S': 0}, 0.8)
        tH.set_probability_dict({'H': 0, 'S': 1}, 0.3)
        tH.set_probability_dict({'H': 1, 'S': 0}, 0.2)
        tH.set_probability_dict({'H': 1, 'S': 1}, 0.7)

        net = BayesianNet()
        net.add_variable(self.L)
        net.add_variable(self.S)
        net.add_variable(self.D)
        net.add_variable(self.H)

        net.add_dependence('L', 'S')
        net.add_dependence('L', 'D')
        net.add_dependence('H', 'S')

        net.add_prob_table('L', tL)
        net.add_prob_table('S', tS)
        net.add_prob_table('D', tD)
        net.add_prob_table('H', tH)

        jtree = JunctionTree(dict.fromkeys([self.S, self.D, self.L, self.H]))
        jtree.add_clique(['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('S', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('D', ['S', 'D', 'L'])
        jtree.set_variable_chosen_clique('L', ['S', 'D', 'L'])

        jtree.add_clique(['S', 'H'])
        jtree.set_variable_chosen_clique('H', ['H', 'S'])

        jtree.add_separator(['S'])
        jtree.add_link(['S', 'H'], ['S'])
        jtree.add_link(['S', 'D', 'L'], ['S'])

        jtree.initialize_tables(net)

        jtree.add_evidence('L', 0)
        jtree.add_evidence('H', 1)
        jtree.sum_propagate()

        STable = jtree.calculate_variable_probability('S')
        DTable = jtree.calculate_variable_probability('D')
        LTable = jtree.calculate_variable_probability('L')
        HTable = jtree.calculate_variable_probability('H')

        self.assertAlmostEqual(round(STable.get_prob(0), 4), 0.9604)
        self.assertAlmostEqual(round(STable.get_prob(1), 4), 0.0396)

        self.assertAlmostEqual(round(DTable.get_prob(0), 4), 0.9819)
        self.assertAlmostEqual(round(DTable.get_prob(1), 4), 0.0181)

        self.assertAlmostEqual(round(LTable.get_prob(0), 4), 1)
        self.assertAlmostEqual(round(LTable.get_prob(1), 4), 0)

        self.assertAlmostEqual(round(HTable.get_prob(0), 4), 0)
        self.assertAlmostEqual(round(HTable.get_prob(1), 4), 1)

if __name__ == '__main__':
    unittest.main()



