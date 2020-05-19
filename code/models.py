#
# This file contains functions that build a bayesian net and its junction tree
# Each function returns a tuple (BayesianNet, JunctionTree)
#
from bayes_nets import *


def build_cancer():
    MC = Variable('MC', 'Metastatic Cancer', ['Present', 'Absent'])
    S = Variable('S', 'Serum Calcium', ['Increased', 'Not increased'])
    T = Variable('T', 'Brain Tumor', ['Present', 'Absent'])
    C = Variable('C', 'Coma', ['Present', 'Absent'])
    H = Variable('H', 'Severe Headaches', ['Present', 'Absent'])

    tMC = BeliefTable([MC])
    tS = BeliefTable([S, MC])
    tT = BeliefTable([T, MC])
    tC = BeliefTable([C, S, T])
    tH = BeliefTable([H, T])

    tMC.set_probability_dict({'MC': 'Absent'}, 0.8)
    tMC.set_probability_dict({'MC': 'Present'}, 0.2)

    tS.set_probability_dict({'S': 'Not increased', 'MC': 'Absent'}, 0.8)
    tS.set_probability_dict({'S': 'Not increased', 'MC': 'Present'}, 0.2)
    tS.set_probability_dict({'S': 'Increased', 'MC': 'Absent'}, 0.2)
    tS.set_probability_dict({'S': 'Increased', 'MC': 'Present'}, 0.8)

    tT.set_probability_dict({'T': 'Absent', 'MC': 'Absent'}, 0.95)
    tT.set_probability_dict({'T': 'Absent', 'MC': 'Present'}, 0.8)
    tT.set_probability_dict({'T': 'Present', 'MC': 'Absent'}, 0.05)
    tT.set_probability_dict({'T': 'Present', 'MC': 'Present'}, 0.2)

    tC.set_probability_dict({'C': 'Absent', 'S': 'Not increased', 'T': 'Absent'}, 0.95)
    tC.set_probability_dict({'C': 'Absent', 'S': 'Not increased', 'T': 'Present'}, 0.2)
    tC.set_probability_dict({'C': 'Absent', 'S': 'Increased', 'T': 'Absent'}, 0.2)
    tC.set_probability_dict({'C': 'Absent', 'S': 'Increased', 'T': 'Present'}, 0.2)
    tC.set_probability_dict({'C': 'Present', 'S': 'Not increased', 'T': 'Absent'}, 0.05)
    tC.set_probability_dict({'C': 'Present', 'S': 'Not increased', 'T': 'Present'}, 0.8)
    tC.set_probability_dict({'C': 'Present', 'S': 'Increased', 'T': 'Absent'}, 0.8)
    tC.set_probability_dict({'C': 'Present', 'S': 'Increased', 'T': 'Present'}, 0.8)

    tH.set_probability_dict({'H': 'Absent', 'T': 'Absent'}, 0.4)
    tH.set_probability_dict({'H': 'Absent', 'T': 'Present'}, 0.2)
    tH.set_probability_dict({'H': 'Present', 'T': 'Absent'}, 0.6)
    tH.set_probability_dict({'H': 'Present', 'T': 'Present'}, 0.8)

    net = BayesianNet()
    net.add_variable(MC)
    net.add_variable(S)
    net.add_variable(T)
    net.add_variable(C)
    net.add_variable(H)

    net.add_dependence('S', 'MC')
    net.add_dependence('T', 'MC')
    net.add_dependence('C', 'S')
    net.add_dependence('C', 'T')
    net.add_dependence('H', 'T')

    net.add_prob_table('MC', tMC)
    net.add_prob_table('S', tS)
    net.add_prob_table('T', tT)
    net.add_prob_table('C', tC)
    net.add_prob_table('H', tH)

    jtree = JunctionTree([MC, S, T, C, H])

    jtree.add_clique(['T', 'S', 'C'])
    jtree.add_clique(['T', 'S', 'MC'])
    jtree.add_clique(['T', 'H'])


    jtree.connect_cliques(['T', 'S', 'MC'], ['T', 'S', 'C'])
    jtree.connect_cliques(['T', 'S', 'MC'], ['T', 'H'])


    jtree.set_variable_chosen_clique('MC', ['T', 'S', 'MC'])
    jtree.set_variable_chosen_clique('S', ['T', 'S', 'MC'])
    jtree.set_variable_chosen_clique('T', ['T', 'S', 'MC'])
    jtree.set_variable_chosen_clique('C', ['T', 'S', 'C'])
    jtree.set_variable_chosen_clique('H', ['T', 'H'])

    return net, jtree

def build_fire():
    S = Variable('S', 'Smoke', ['true', 'false'])
    F = Variable('F', 'Fire', ['true', 'false'])
    A = Variable('A', 'Alarm', ['true', 'false'])
    T = Variable('T', 'Tampering', ['true', 'false'])
    L = Variable('L', 'Leaving', ['true', 'false'])
    R = Variable('R', 'Report', ['true', 'false'])

    tS = BeliefTable([S, F])
    tF = BeliefTable([F])
    tA = BeliefTable([A, F, T])
    tT = BeliefTable([T])
    tL = BeliefTable([A, L])
    tR = BeliefTable([R, L])

    # Fill tables
    tS.set_probability_dict({'F': 'true', 'S': 'true'}, 0.9)
    tS.set_probability_dict({'F': 'true', 'S': 'false'}, 0.1)
    tS.set_probability_dict({'F': 'false', 'S': 'true'}, 0.01)
    tS.set_probability_dict({'F': 'false', 'S': 'false'}, 0.99)

    tF.set_probability_dict({'F': 'true'}, 0.01)
    tF.set_probability_dict({'F': 'false'}, 0.99)

    tA.set_probability_dict({'F': 'true', 'T': 'true', 'A': 'true'}, 0.5)
    tA.set_probability_dict({'F': 'true', 'T': 'true', 'A': 'false'}, 0.5)
    tA.set_probability_dict({'F': 'true', 'T': 'false', 'A': 'true'}, 0.99)
    tA.set_probability_dict({'F': 'true', 'T': 'false', 'A': 'false'}, 0.01)
    tA.set_probability_dict({'F': 'false', 'T': 'true', 'A': 'true'}, 0.85)
    tA.set_probability_dict({'F': 'false', 'T': 'true', 'A': 'false'}, 0.15)
    tA.set_probability_dict({'F': 'false', 'T': 'false', 'A': 'true'}, 0.0001)
    tA.set_probability_dict({'F': 'false', 'T': 'false', 'A': 'false'}, 0.9999)

    tT.set_probability_dict({'T': 'true'}, 0.02)
    tT.set_probability_dict({'T': 'false'}, 0.98)

    tL.set_probability_dict({'A': 'true', 'L': 'true'}, 0.88)
    tL.set_probability_dict({'A': 'true', 'L': 'false'}, 0.12)
    tL.set_probability_dict({'A': 'false', 'L': 'true'}, 0.001)
    tL.set_probability_dict({'A': 'false', 'L': 'false'}, 0.999)

    tR.set_probability_dict({'L': 'true', 'R': 'true'}, 0.75)
    tR.set_probability_dict({'L': 'true', 'R': 'false'}, 0.25)
    tR.set_probability_dict({'L': 'false', 'R': 'true'}, 0.01)
    tR.set_probability_dict({'L': 'false', 'R': 'false'}, 0.99)

    net = BayesianNet()

    net.add_variable(S)
    net.add_variable(F)
    net.add_variable(A)
    net.add_variable(T)
    net.add_variable(L)
    net.add_variable(R)

    net.add_dependence(S, F)
    net.add_dependence(A, F)
    net.add_dependence(A, T)
    net.add_dependence(L, A)
    net.add_dependence(R, L)

    net.add_prob_table(S, tS)
    net.add_prob_table(F, tF)
    net.add_prob_table(A, tA)
    net.add_prob_table(T, tT)
    net.add_prob_table(L, tL)
    net.add_prob_table(R, tR)

    jtree = JunctionTree([S, F, A, T, L, R])

    jtree.add_clique([F, S])
    jtree.add_clique([A, F, T])
    jtree.add_clique([L, A])
    jtree.add_clique([L, R])

    jtree.connect_cliques([F, S], [A, F, T])
    jtree.connect_cliques([A, F, T], [L, A])
    jtree.connect_cliques([L, A], [L, R])

    jtree.set_variable_chosen_clique(S, [S, F])
    jtree.set_variable_chosen_clique(F, [S, F])
    jtree.set_variable_chosen_clique(A, [A, F, T])
    jtree.set_variable_chosen_clique(T, [A, F, T])
    jtree.set_variable_chosen_clique(L, [A, L])
    jtree.set_variable_chosen_clique(R, [L, R])

    return net, jtree


def build_poker():
    possible_hands = ['no', '1 ace', '2 consecutive', '2 of a suit', '2 alike', 'flush', 'straight', '3 alike',
                      'straight flush']
    OH0 = Variable('OH0', 'Opponent\'s initial hand', possible_hands)
    OH1 = Variable('OH1', 'Opponent\'s hand after one round of discarding', possible_hands)
    OH2 = Variable('OH2', 'Opponent\'s hand after 2 rounds of discarding',
                   ['no', '1 ace', '2 consecutive', '2 alike', 'flush', 'straight', '3 alike',
                    'straight flush']
                   )
    FC = Variable('FC', 'Number of cards discarded by opp during first round of discarding',
                  ['0 changed', '1 changed', '2 changed', '3 changed'])
    SC = Variable('SC', 'Number of cards discarded by opp during second round of discarding',
                  ['0 changed', '1 changed', '2 changed'])
    MH = Variable('MH', 'My hand', ['no', '1 ace', '2 consecutive', '2 alike', 'flush', 'straight', '3 alike',
                      'straight flush'])
    BH = Variable('BH', 'Who has the best hand', ['myhand', 'opponent'])

    tOH0 = BeliefTable([OH0])
    tOH1 = BeliefTable([OH0, FC, OH1])
    tFC = BeliefTable([OH0, FC])
    tOH2 = BeliefTable([OH1, SC, OH2])
    tSC = BeliefTable([OH1, SC])
    tMH = BeliefTable([MH])
    tBH = BeliefTable([MH, OH2, BH])

    # Fill tables(good luck)
    tMH.set_probability_dict({'MH': slice(None)}, (0.125, 0.125, 0.125 ,0.125, 0.125, 0.125, 0.125, 0.125))

    tOH0.set_probability_dict({'OH0': slice(None)},
                              (0.1672, 0.0445, 0.0635, 0.4659, 0.1694, 0.0494, 0.0353, 0.0024, 0.0024))

    tFC.set_probability_dict({'OH0': 'no', 'FC': slice(None)}, (0, 0, 0, 1))
    tFC.set_probability_dict({'OH0': '1 ace', 'FC': slice(None)}, (0, 0, 1, 0) )
    tFC.set_probability_dict({'OH0': '2 consecutive', 'FC': slice(None)}, (0, 1, 0, 0))
    tFC.set_probability_dict({'OH0': '2 of a suit', 'FC': slice(None)}, (0, 1, 0, 0))
    tFC.set_probability_dict({'OH0': '2 alike', 'FC': slice(None)}, (0, 1, 0, 0))
    tFC.set_probability_dict({'OH0': 'flush', 'FC': slice(None)}, (1, 0, 0, 0))
    tFC.set_probability_dict({'OH0': 'straight', 'FC': slice(None)}, (1, 0, 0, 0))
    tFC.set_probability_dict({'OH0': '3 alike', 'FC': slice(None)}, (1, 0, 0, 0))
    tFC.set_probability_dict({'OH0': 'straight flush', 'FC': slice(None)}, (1, 0, 0, 0))

    tSC.set_probability_dict({'OH1': 'no', 'SC': slice(None)}, (0, 0, 1))
    tSC.set_probability_dict({'OH1': '1 ace', 'SC': slice(None)}, (0, 0, 1) )
    tSC.set_probability_dict({'OH1': '2 consecutive', 'SC': slice(None)}, (0, 1, 0))
    tSC.set_probability_dict({'OH1': '2 of a suit', 'SC': slice(None)}, (0, 1, 0))
    tSC.set_probability_dict({'OH1': '2 alike', 'SC': slice(None)}, (0, 1, 0))
    tSC.set_probability_dict({'OH1': 'flush', 'SC': slice(None)}, (1, 0, 0))
    tSC.set_probability_dict({'OH1': 'straight', 'SC': slice(None)}, (1, 0, 0))
    tSC.set_probability_dict({'OH1': '3 alike', 'SC': slice(None)}, (1, 0, 0))
    tSC.set_probability_dict({'OH1': 'straight flush', 'SC': slice(None)}, (1, 0, 0))

    tBH.set_probability_dict({'MH': 'no', 'OH2': 'no', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': 'no', 'OH2': '1 ace', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'no', 'OH2': '2 consecutive', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'no', 'OH2': '2 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'no', 'OH2': 'flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'no', 'OH2': 'straight', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'no', 'OH2': '3 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'no', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': '1 ace', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': '2 consecutive', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': '2 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': 'flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': 'straight', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': '3 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '1 ace', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': '1 ace', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': '2 consecutive', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': '2 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': 'flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': 'straight', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': '3 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 consecutive', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': '1 ace', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': '2 consecutive', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': '2 alike', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': 'flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': 'straight', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': '3 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '2 alike', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': '1 ace', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': '2 consecutive', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': '2 alike', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': 'flush', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': 'straight', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': '3 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'flush', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': '1 ace', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': '2 consecutive', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': '2 alike', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': 'flush', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': 'straight', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': '3 alike', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'straight', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': '1 ace', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': '2 consecutive', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': '2 alike', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': 'flush', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': 'straight', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': '3 alike', 'BH': slice(None)}, (0.5,0.5))
    tBH.set_probability_dict({'MH': '3 alike', 'OH2': 'straight flush', 'BH': slice(None)}, (0,1))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': 'no', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': '1 ace', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': '2 consecutive', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': '2 alike', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': 'flush', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': 'straight', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': '3 alike', 'BH': slice(None)}, (1,0))
    tBH.set_probability_dict({'MH': 'straight flush', 'OH2': 'straight flush', 'BH': slice(None)}, (0.5,0.5))

    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': 'no', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': '1 ace', 'OH1': slice(None)}, (0, 1, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': '2 consecutive', 'OH1': slice(None)}, (0, 0, 1, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': '2 of a suit', 'OH1': slice(None)}, (0, 0, 0, 1, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': '2 alike', 'OH1': slice(None)}, (0, 0, 0, 0, 1, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': 'flush', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 1, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': 'straight', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0, 1, 0, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': '3 alike', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0, 0, 1, 0))
    tOH1.set_probability_dict({'FC': '0 changed', 'OH0': 'straight flush', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0, 0, 0, 1))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': 'no', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': '1 ace', 'OH1': slice(None)}, (0, 1, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': '2 consecutive', 'OH1': slice(None)}, (0, 0, 0.347, 0.3674, 0.1224, 0, 0.1632, 0, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': '2 of a suit', 'OH1': slice(None)}, (0, 0, 0, 0.6224, 0.1224, 0.2143, 0.0307, 0, 0.0102))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': '2 alike', 'OH1': slice(None)}, (0, 0, 0, 0, 0.9592, 0, 0, 0.0408, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': 'flush', 'OH1': slice(None)}, (0, 0, 1, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': 'straight', 'OH1': slice(None)}, (0, 0, 0, 1, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': '3 alike', 'OH1': slice(None)}, (0, 0, 0, 0, 1, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '1 changed', 'OH0': 'straight flush', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0.5, 0.5, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': 'no', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': '1 ace', 'OH1': slice(None)}, (0, 0.1814, 0.0681, 0.4796, 0.1738, 0.0536, 0.0383, 0.0026, 0.0026))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': '2 consecutive', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': '2 of a suit', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': '2 alike', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': 'flush', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': 'straight', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': '3 alike', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '2 changed', 'OH0': 'straight flush', 'OH1': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': 'no', 'OH1': slice(None)}, (0.1583, 0.0534, 0.0635, 0.4659, 0.1694, 0.0494, 0.0353, 0.0024, 0.0024))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': '1 ace', 'OH1': slice(None)}, (0, 1, 0, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': '2 consecutive', 'OH1': slice(None)}, (0, 0, 1, 0, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': '2 of a suit', 'OH1': slice(None)}, (0, 0, 0, 1, 0, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': '2 alike', 'OH1': slice(None)}, (0, 0, 0, 0, 1, 0, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': 'flush', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 1, 0, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': 'straight', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0, 1, 0, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': '3 alike', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0, 0, 1, 0))
    tOH1.set_probability_dict({'FC': '3 changed', 'OH0': 'straight flush', 'OH1': slice(None)}, (0, 0, 0, 0, 0, 0, 0, 0, 1))

    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': 'no', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': '1 ace', 'OH2': slice(None)}, (0, 1, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': '2 consecutive', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': '2 of a suit', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': '2 alike', 'OH2': slice(None)}, (0, 0, 0.9231, 0.0769, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': 'flush', 'OH2': slice(None)}, (0, 0, 0, 0, 1, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': 'straight', 'OH2': slice(None)}, (0, 0, 0, 0, 0, 1, 0, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': '3 alike', 'OH2': slice(None)}, (0, 0, 0, 0, 0, 0, 1, 0))
    tOH2.set_probability_dict({'SC': '0 changed', 'OH1': 'straight flush', 'OH2': slice(None)}, (0, 0, 0, 0, 0, 0, 0, 1))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': 'no', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': '1 ace', 'OH2': slice(None)}, (0, 1, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': '2 consecutive', 'OH2': slice(None)}, (0.5903, 0.1181, 0.1154, 0.0096, 0, 0.1666, 0, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': '2 of a suit', 'OH2': slice(None)}, (0.5121, 0.1024, 0.1154, 0.0096, 0.2188, 0.0313, 0, 0.0104))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': '2 alike', 'OH2': slice(None)}, (0, 0, 0.8838, 0.0736, 0, 0, 0.0426, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': 'flush', 'OH2': slice(None)}, (0, 0, 0, 0, 1, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': 'straight', 'OH2': slice(None)}, (0, 0, 0, 0, 0, 1, 0, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': '3 alike', 'OH2': slice(None)}, (0, 0, 0, 0, 0, 0, 1, 0))
    tOH2.set_probability_dict({'SC': '1 changed', 'OH1': 'straight flush', 'OH2': slice(None)}, (0, 0, 0, 0, 0, 0, 0, 1))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': 'no', 'OH2': slice(None)}, (0.5613, 0.1570, 0.1757, 0.0055, 0.0559, 0.0392, 0.0027, 0.0027))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': '1 ace', 'OH2': slice(None)}, (0, 0.7183, 0.0667, 0.1145, 0.0559, 0.0392, 0.0027, 0.0027))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': '2 consecutive', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': '2 of a suit', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': '2 alike', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': 'flush', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': 'straight', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': '3 alike', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))
    tOH2.set_probability_dict({'SC': '2 changed', 'OH1': 'straight flush', 'OH2': slice(None)}, (1, 0, 0, 0, 0, 0, 0, 0))

    #################
    net = BayesianNet()
    net.add_variable(OH0)
    net.add_variable(OH1)
    net.add_variable(OH2)
    net.add_variable(FC)
    net.add_variable(SC)
    net.add_variable(MH)
    net.add_variable(BH)

    net.add_dependence(FC, OH0)
    net.add_dependence(OH1, OH0)
    net.add_dependence(OH1, FC)
    net.add_dependence(SC, OH1)
    net.add_dependence(OH2, OH1)
    net.add_dependence(OH2, SC)
    net.add_dependence(BH, MH)
    net.add_dependence(BH, OH2)

    net.add_prob_table(OH0, tOH0)
    net.add_prob_table(OH1, tOH1)
    net.add_prob_table(OH2, tOH2)
    net.add_prob_table(FC, tFC)
    net.add_prob_table(SC, tSC)
    net.add_prob_table(MH, tMH)
    net.add_prob_table(BH, tBH)

    jtree = JunctionTree([OH0, OH1, OH2, FC, SC, MH, BH])

    jtree.add_clique([OH1, OH0, FC])
    jtree.add_clique([OH2, OH1, SC])
    jtree.add_clique([OH2, BH, MH])

    jtree.connect_cliques([OH1, OH0, FC], [OH1, OH2, SC])
    jtree.connect_cliques([OH1, OH2, SC], [OH2, BH, MH])

    jtree.set_variable_chosen_clique(OH0, [OH1, OH0, FC])
    jtree.set_variable_chosen_clique(FC, [OH1, OH0, FC])
    jtree.set_variable_chosen_clique(OH1, [OH1, OH0, FC])
    jtree.set_variable_chosen_clique(SC, [OH2, OH1, SC])
    jtree.set_variable_chosen_clique(OH2, [OH2, OH1, SC])
    jtree.set_variable_chosen_clique(MH, [OH2, BH, MH])
    jtree.set_variable_chosen_clique(BH, [OH2, BH, MH])

    return net, jtree



def build_monty():
    P = Variable('P', 'Prize', ['door1', 'door2', 'door3'])
    F = Variable('F', 'First Choice', ['door1', 'door2', 'door3'])
    M = Variable('M', 'Monty Opens', ['door1', 'door2', 'door3'])

    tP = BeliefTable([P])
    tF = BeliefTable([F])
    tM = BeliefTable([P, F, M])

    # Fill tables
    tP.set_probability_dict({'P': 'door1'}, 0.3333)
    tP.set_probability_dict({'P': 'door2'}, 0.3333)
    tP.set_probability_dict({'P': 'door3'}, 0.3333)

    tF.set_probability_dict({'F': 'door1'}, 0.3333)
    tF.set_probability_dict({'F': 'door2'}, 0.3333)
    tF.set_probability_dict({'F': 'door3'}, 0.3333)

    tM.set_probability_dict({'P': 'door1', 'F': 'door1', 'M': 'door1'}, 0)
    tM.set_probability_dict({'P': 'door1', 'F': 'door1', 'M': 'door2'}, 0.5)
    tM.set_probability_dict({'P': 'door1', 'F': 'door1', 'M': 'door3'}, 0.5)
    tM.set_probability_dict({'P': 'door1', 'F': 'door2', 'M': 'door1'}, 0)
    tM.set_probability_dict({'P': 'door1', 'F': 'door2', 'M': 'door2'}, 0)
    tM.set_probability_dict({'P': 'door1', 'F': 'door2', 'M': 'door3'}, 1)
    tM.set_probability_dict({'P': 'door1', 'F': 'door3', 'M': 'door1'}, 0)
    tM.set_probability_dict({'P': 'door1', 'F': 'door3', 'M': 'door2'}, 1)
    tM.set_probability_dict({'P': 'door1', 'F': 'door3', 'M': 'door3'}, 0)
    tM.set_probability_dict({'P': 'door2', 'F': 'door1', 'M': 'door1'}, 0)
    tM.set_probability_dict({'P': 'door2', 'F': 'door1', 'M': 'door2'}, 0)
    tM.set_probability_dict({'P': 'door2', 'F': 'door1', 'M': 'door3'}, 1)
    tM.set_probability_dict({'P': 'door2', 'F': 'door2', 'M': 'door1'}, 0.5)
    tM.set_probability_dict({'P': 'door2', 'F': 'door2', 'M': 'door2'}, 0)
    tM.set_probability_dict({'P': 'door2', 'F': 'door2', 'M': 'door3'}, 0.5)
    tM.set_probability_dict({'P': 'door2', 'F': 'door3', 'M': 'door1'}, 1)
    tM.set_probability_dict({'P': 'door2', 'F': 'door3', 'M': 'door2'}, 0)
    tM.set_probability_dict({'P': 'door2', 'F': 'door3', 'M': 'door3'}, 0)
    tM.set_probability_dict({'P': 'door3', 'F': 'door1', 'M': 'door1'}, 0)
    tM.set_probability_dict({'P': 'door3', 'F': 'door1', 'M': 'door2'}, 1)
    tM.set_probability_dict({'P': 'door3', 'F': 'door1', 'M': 'door3'}, 0)
    tM.set_probability_dict({'P': 'door3', 'F': 'door2', 'M': 'door1'}, 1)
    tM.set_probability_dict({'P': 'door3', 'F': 'door2', 'M': 'door2'}, 0)
    tM.set_probability_dict({'P': 'door3', 'F': 'door2', 'M': 'door3'}, 0)
    tM.set_probability_dict({'P': 'door3', 'F': 'door3', 'M': 'door1'}, 0.5)
    tM.set_probability_dict({'P': 'door3', 'F': 'door3', 'M': 'door2'}, 0.5)
    tM.set_probability_dict({'P': 'door3', 'F': 'door3', 'M': 'door3'}, 0)

    net = BayesianNet()
    net.add_variable(P)
    net.add_variable(F)
    net.add_variable(M)

    net.add_dependence(M, F)
    net.add_dependence(M, P)

    net.add_prob_table(P, tP)
    net.add_prob_table(F, tF)
    net.add_prob_table(M, tM)

    jtree = JunctionTree([P, F, M])

    jtree.add_clique([P, F, M])
    jtree.set_variable_chosen_clique(P, [P, F, M])
    jtree.set_variable_chosen_clique(F, [P, F, M])
    jtree.set_variable_chosen_clique(M, [P, F, M])

    return net, jtree


def build_chestclinic():
    A = Variable('A', "Visit to Asia?", ['yes', 'no'])
    T = Variable('T', "Tuberculosis?", ['yes', 'no'])
    L = Variable('L', "Lung cancer?", ['yes', 'no'])
    E = Variable('E', "Tuberculosis or cancer?", ['yes', 'no'])
    S = Variable('S', "Smoker?", ['yes', 'no'])
    B = Variable('B', "Bronchitis?", ['yes', 'no'])
    D = Variable('D', "Dyspnoea?", ['yes', 'no'])
    X = Variable('X', "Positive X-ray?", ['yes', 'no'])

    tA = BeliefTable([A])
    tT = BeliefTable([A, T])
    tL = BeliefTable([S, L])
    tE = BeliefTable([E, L, T])
    tS = BeliefTable([S])
    tB = BeliefTable([S, B])
    tD = BeliefTable([E, D, B])
    tX = BeliefTable([E, X])

    # Fill tables
    tA.set_probability_dict({'A': 'yes'}, 0.01)
    tA.set_probability_dict({'A': 'no'}, 0.99)

    tT.set_probability_dict({'A': 'yes', 'T': 'yes'}, 0.05)
    tT.set_probability_dict({'A': 'yes', 'T': 'no'}, 0.95)
    tT.set_probability_dict({'A': 'no', 'T': 'yes'}, 0.01)
    tT.set_probability_dict({'A': 'no', 'T': 'no'}, 0.99)

    tL.set_probability_dict({'S': 'yes', 'L': 'yes'}, 0.1)
    tL.set_probability_dict({'S': 'yes', 'L': 'no'}, 0.9)
    tL.set_probability_dict({'S': 'no', 'L': 'yes'}, 0.01)
    tL.set_probability_dict({'S': 'no', 'L': 'no'}, 0.99)

    tE.set_probability_dict({'T': 'yes', 'L': 'yes', 'E': 'yes'}, 1)
    tE.set_probability_dict({'T': 'yes', 'L': 'yes', 'E': 'no'}, 0)
    tE.set_probability_dict({'T': 'yes', 'L': 'no', 'E': 'yes'}, 1)
    tE.set_probability_dict({'T': 'yes', 'L': 'no', 'E': 'no'}, 0)
    tE.set_probability_dict({'T': 'no', 'L': 'yes', 'E': 'yes'}, 1)
    tE.set_probability_dict({'T': 'no', 'L': 'yes', 'E': 'no'}, 0)
    tE.set_probability_dict({'T': 'no', 'L': 'no', 'E': 'yes'}, 0)
    tE.set_probability_dict({'T': 'no', 'L': 'no', 'E': 'no'}, 1)

    tS.set_probability_dict({'S': 'yes'}, 0.5)
    tS.set_probability_dict({'S': 'no'}, 0.5)

    tB.set_probability_dict({'S': 'yes', 'B': 'yes'}, 0.6)
    tB.set_probability_dict({'S': 'yes', 'B': 'no'}, 0.4)
    tB.set_probability_dict({'S': 'no', 'B': 'yes'}, 0.3)
    tB.set_probability_dict({'S': 'no', 'B': 'no'}, 0.7)

    tD.set_probability_dict({'B': 'yes', 'E': 'yes', 'D': 'yes'}, 0.9)
    tD.set_probability_dict({'B': 'yes', 'E': 'yes', 'D': 'no'}, 0.1)
    tD.set_probability_dict({'B': 'yes', 'E': 'no', 'D': 'yes'}, 0.8)
    tD.set_probability_dict({'B': 'yes', 'E': 'no', 'D': 'no'}, 0.2)
    tD.set_probability_dict({'B': 'no', 'E': 'yes', 'D': 'yes'}, 0.7)
    tD.set_probability_dict({'B': 'no', 'E': 'yes', 'D': 'no'}, 0.3)
    tD.set_probability_dict({'B': 'no', 'E': 'no', 'D': 'yes'}, 0.1)
    tD.set_probability_dict({'B': 'no', 'E': 'no', 'D': 'no'}, 0.9)

    tX.set_probability_dict({'E': 'yes', 'X': 'yes'}, 0.98)
    tX.set_probability_dict({'E': 'yes', 'X': 'no'}, 0.02)
    tX.set_probability_dict({'E': 'no', 'X': 'yes'}, 0.05)
    tX.set_probability_dict({'E': 'no', 'X': 'no'}, 0.95)

    net = BayesianNet()

    net.add_variable(A)
    net.add_variable(T)
    net.add_variable(L)
    net.add_variable(E)
    net.add_variable(S)
    net.add_variable(B)
    net.add_variable(D)
    net.add_variable(X)

    net.add_dependence(T, A)
    net.add_dependence(L, S)
    net.add_dependence(B, S)
    net.add_dependence(E, L)
    net.add_dependence(E, T)
    net.add_dependence(D, E)
    net.add_dependence(D, B)
    net.add_dependence(X, E)

    net.add_prob_table(A, tA)
    net.add_prob_table(T, tT)
    net.add_prob_table(L, tL)
    net.add_prob_table(E, tE)
    net.add_prob_table(S, tS)
    net.add_prob_table(B, tB)
    net.add_prob_table(D, tD)
    net.add_prob_table(X, tX)

    jtree = JunctionTree([A, T, L, E, S, B, D, X])

    jtree.add_clique([A, T])
    jtree.add_clique([E, L, T])
    jtree.add_clique([E, S, L])
    jtree.add_clique([E, B, S])
    jtree.add_clique([E, B, D])
    jtree.add_clique([E, X])


    jtree.connect_cliques([A, T], [E, L, T])
    jtree.connect_cliques([E, S, L], [E, L, T])
    jtree.connect_cliques([E, S, B], [E, L, S])
    jtree.connect_cliques([E, S, B], [E, B, D])
    jtree.connect_cliques([E, B, D], [E, X])

    jtree.set_variable_chosen_clique(A, [A, T])
    jtree.set_variable_chosen_clique(T, [A, T])
    jtree.set_variable_chosen_clique(L, [E, S, L])
    jtree.set_variable_chosen_clique(E, [E, L, T])
    jtree.set_variable_chosen_clique(S, [E, B, S])
    jtree.set_variable_chosen_clique(B, [E, B, S])
    jtree.set_variable_chosen_clique(D, [E, B, D])
    jtree.set_variable_chosen_clique(X, [E, X])

    return net, jtree










