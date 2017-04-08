#!/usr/bin/env python3
#
########################################################################
# File: ba10k.py
#  executable: ba10k.py
# Purpose:
#   stderr: not used
#   stdout: optimized transition and emissions using baum welch learning
#
# Author: David Bernick
# History:      dlb 3/12/2017 Created
#
########################################################################


########################################################################
# Main
########################################################################

import sys
from collections import namedtuple
import numpy as np
Node = namedtuple('Node', list('mdi'))
class HMM ():
    def __init__ (self, emits=[], states=[], transition=dict(), emission=dict(), theta=None, sigma = 0.0):
        self.emits = emits
        self.states = states
        self.transition = transition
        self.emission = emission
        self.theta = theta
        self.sigma=sigma
        self.notSigma = 1 - 3*sigma # used in normalizeNode()
    def PrPi (self, path):
        P = 1/len(self.states) # transition from start to first states
        prev = path[0]

        for next in path[1:]:
            P *= self.transition[prev][next]
            prev = next
        return P

    def PrPiEm (self, path, outcome):
        P = self.emission[path[0]][outcome[0]] / len(self.states)  # transition from start to first
        prev = path[0]

        for i in range(1,len(outcome)):
            P *= self.transition[path[i-1]][path[i]] * self.emission[path[i]][outcome[i]]

        return P

    def PrOut (self, outcome, path):
        ''' compute P(outcome | path) '''
        P = 1 # initial value
        prev = path[0]

        for state, emit in zip(path, outcome):
            P *= self.emission[state][emit]
        return P

    def buildTransition(self):
        ''' construct the transition matrix from stdin positioned to read column header'''
        self.states = sys.stdin.readline().rstrip().split()

        for line in sys.stdin:
            if line.startswith('-'):
                break
            vals = line.rstrip().split()
            self.transition[vals[0]] = {state: np.longdouble(val) for state, val in zip(self.states, vals[1:])}

    def buildEmission(self):
        ''' construct the emission matrix from stdin positioned to read column header'''
        self.emits = sys.stdin.readline().rstrip().split()

        for line in sys.stdin:
            if line.startswith('-'):
                break
            vals = line.rstrip().split()
            self.emission[vals[0]] = {emit: np.longdouble(val) for emit, val in zip(self.emits, vals[1:])}

    def viterbi(self, outcome):
        v = [ {s: self.emission[s][outcome[0]] / len(self.states) for s in self.states } ]
        for c in outcome[1:]:
            v.append ( {
                           k: self.emission[k][c] *
                           max( [
                                v[-1][l] * self.transition[l][k] for l in self.states
                                ])
                           for k in self.states
                        } )

        # traceback
        pi = [self.argMax(v[-1])]
        for i in range(len(v)-1,0,-1):
            k = pi[0]
            pi[0:0] = self.argMax ( {l: v[i-1][l] * self.transition[l][k]  for l in self.states})
        return ''.join(pi), max(v[-1].values())

    def argMax (self, d):
        amax = 0.0
        maxKey = None
        for key, value in d.items():
            if value >= amax:
                maxKey = key
                amax = value
        return maxKey

    def forward(self, outcome):

        # equivalent to:  v.append( v[-1] * emit.T[c] ).T @ self.transition )

        # build viterbi graph initially with ( 1/#states ) * emission(x0) for each of the first node
        # assumes an initial start with P=1
        v = [ {s: self.emission[s][outcome[0]] / len(self.states) for s in self.states } ]

        # build the remainder of the viterbi graph

        for c in outcome[1:]:
            v.append ( {
                           k: self.emission[k][c] *
                           sum( [
                                v[-1][l] * self.transition[l][k] for l in self.states
                                ])
                           for k in self.states} )

        return v
    def backwardOld(self, outcome):

        v = [ {s: self.emission[s][outcome[-1]] for s in self.states } ]

        # build the remainder of the viterbi graph

        revOutcome = outcome[-2::-1]
        for c in revOutcome:
            v.insert(0, {
                           k: self.emission[k][c] *
                           sum( [
                                v[0][l] * self.transition[k][l] for l in self.states # reversed transitions
                                ])
                           for k in self.states} )

        return v

    def backward(self, outcome):

        v = [ {s: 1 for s in self.states } ]

        # build the remainder of the viterbi graph

        for c in reversed(outcome[1:]):
            v.insert(0, {
                           k:
                           sum( [ self.emission[l][c] *
                                v[0][l] * self.transition[k][l] for l in self.states # reversed transitions
                                ])
                           for k in self.states} )

        return v

    def buildPiStar(self, outcome):
        leftViterbi = self.forward(outcome)
        rightViterbi = self.backward(outcome)

        invPrSequence = 1/sum(leftViterbi[-1].values())
        piStar = []
        for i in range(len(outcome)):
            piStar.append(
                {l: leftViterbi[i][l] * rightViterbi[i][l] * invPrSequence
                 for l in self.states}
            )

        return piStar


    def buildPiStarStar (self, outcome):
        leftViterbi = self.forward(outcome)
        rightViterbi = self.backward(outcome)

        invPrSequence = sum(leftViterbi[-1].values())

        piStarStar = []
        for i in range(len(outcome)-1):

                piStarStar.append (
                    {l: {k:
                        leftViterbi[i][l] *
                        rightViterbi[i+1][k] *
                        self.emission[k][outcome[i+1]] *
                        self.transition[l][k] * invPrSequence
                        for k in self.states}
                    for l in self.states}
                )
        return piStarStar

    def maximizePi(self, outcome):

        piStar = self.buildPiStar(outcome)

        piStarStar = self.buildPiStarStar(outcome)

        newEmissions = {k: {e: sum( [a[k] for a,c in zip(piStar, outcome) if c == e]) for e in self.emits} for k in self.states}
        newTransition = {l: {k: sum ( [ steps[l][k] for steps in piStarStar] ) for k in self.states} for l in self.states}

        self.emission = {k: self._normalize(newEmissions[k]) for k in self.states}
        self.transition = {k: self._normalize(newTransition[k]) for k in self.states}

    def printPi(self, piStar, piStarStar):
        print('\t'.join(self.states))
        for step in piStar:
            for s in self.states:
                print("{0:<.3f}\t".format(step [s]), end='')
            print()

        print("\t{}".format('\t'.join([str(i) for i in range(1, len(piStarStar)+1)])))

        for l in self.states:
            for k in self.states:
                print(l + k, sep='', end='')
                for step in piStarStar:
                    print("\t{0:<.3f}".format(step [l][k]), end='')
                print()

    def writeVgraph (self, v):
        for i in range(len(v)):
            print ("\t{}".format(i), end = '')
        print ()
        for s in self.states:
            print(s, end='')
            for i in range (len(v)):
                print ("\t{}".format(v[i][s]), end = '')
            print ()

    def buildMatrices (self, mAlign):

        # add start and end symbols to each sequence
        ali = ['^' + seq + '^' for seq in mAlign ]

        depth = len(ali)
        width = len(ali[0])

        begin = 0 # set up as a non-insert as a result of adding S
        p = 1 # the first column to check
        gapThreshold = self.theta * depth
        stateSets = []
        emitSets = []
        while p < width: #keep gathering until we have another non-insert column
            gaps = len([seq for seq in ali if seq[p] == '-'])
            if gaps < gapThreshold: # we have a non insert column

                # build a column worth of sequence. It starts and ends with non-insert columns

                T,e = self.buildTransitionStep([seq[begin:p + 1] for seq in ali])

                stateSets.append(T)
                emitSets.append(e)
                begin = p
            p += 1

        self.states = ['S', ('i',0) ]
        for i in range (1, len(stateSets)):
            self.states += [ (s,i) for s in 'mdi' ]
        self.states += ['E']

        A = {d:{state:0. for state in self.states} for d in self.states}
        emissions = {d:{emit:0. for emit in self.emits} for d in self.states}

        for p in range(1, len(stateSets)-1):
            for j,k in enumerate('mdi'):
                A[(k, p)][('m', p + 1)] = stateSets[p][j].m # state set tuples are ( m, d, i )
                A[(k, p)][('d', p + 1)] = stateSets[p][j].d
                A[(k, p)][('i', p)]     = stateSets[p][j].i
        last = p+1

        # fixup beginning transitions
        A['S'][('m', 1)] = stateSets[0].m.m  # transitions from start
        A['S'][('d', 1)] = stateSets[0].m.d
        A['S'][('i', 0)] = stateSets[0].m.i

        A[('i', 0)][('m', 1)] = stateSets[0].i.m  # transitions from i0
        A[('i', 0)][('d', 1)] = stateSets[0].i.d
        A[('i', 0)][('i', 0)] = stateSets[0].i.i

        # transitions to End
        # reapportion weight to next d over m and i
        for j, k in enumerate('mdi'):
            v = stateSets[last][j].m / (stateSets[last][j].m + stateSets[last][j].i)
            A[(k, last)]['E'] = v
            A[(k, last)][('i', last)] = 1 - v

        for p in range(1, len(emitSets) ):
            emissions[('i', p)] = emitSets[p]['i']
            emissions[('m', p)] = emitSets[p]['m']
        emissions[('i', 0)] = emitSets[0]['i']

        # add alias links to make addressing of the Start node more uniform. It is treated as an set of 0 rank cells.
        A[('m', 0)] = {('m', 1):  A['S'][('m', 1)], ('d', 1): A['S'][('d', 1)] }
        A[('d', 0)] = {('m', 1):  A['S'][('m', 1)], ('d', 1): A['S'][('d', 1)] }

        self.transition = A
        self.emission = emissions

    def buildTransitionStep (self, ali):

        ''' construct a set of M, I, D transition probs from an alignment slice '''
        # alignment matrix has a first and last column that is subTheta
        # any internal columns are theta+

        chars = self.emits + list('^')
        gChars = chars + list('-')

        emissions = {}
        emissions['m'] = {s:0. for s in self.emits}
        emissions['i'] = {s:0. for s in self.emits}

        l = {c:0 for c in gChars} # count matches and deletes in l
        lk = {c+d:0 for c in gChars for d in gChars} # count transitions from l to k
        lI = l.copy() # count entrances to insert
        Ik = l.copy() # count exits from insert

        for seq in ali:
            l[seq[0]] += 1
            if all( [c is '-' for c in seq[1:-1]] ): # check if there is no actual insertion data
                lk[seq[0]+seq[-1]] += 1

            else:
                lI[seq[0]] += 1  # counts of transition kinds to Insert
                Ik[seq[-1]] += 1 # counts of trahsition kinds out of Insert
                for c in seq[1:-1]:
                    if c in self.emits:
                        emissions['i'][c] += 1

        T = Node(   m = self._normalizeNode(
                    sum([lk[c + d] for c in chars for d in chars]),  # m
                    sum([lk[c + '-'] for c in chars]), # d
                    sum(lI.values()) - lI['-']  # i
                    ),
                    d = self._normalizeNode(
                    sum([lk['-' + c] for c in chars]),  # m
                    lk['--'], # d
                    lI['-']   # i
                    ),
                    i = self._normalizeNode(
                    sum(lI.values()) - Ik['-'],  # m
                    Ik['-'],  # d
                    sum(emissions['i'].values()) - sum(lI.values())  # i
                    )
        )

        emissions['m'] = self._normalize( {emit:l[emit] for emit in self.emits } )
        emissions['i'] = self._normalize ( emissions['i'])

        return T, emissions

    def _normalize(self, node):
        l = len(node)
        if any(node.values()): # this is not a zero node
            t = sum(node.values())
            k = {n: np.longdouble(val / t) for n, val in node.items()}
            notSigma = 1 - self.sigma*l
            return {n: val * notSigma + self.sigma for n, val in k.items()}
        else:
            return {n:1/l for n in node.keys()}

    def _normalizeNode(self, *node):

        if any(node):  # this is not a zero node
            return Node ( *(val/sum(node) * self.notSigma + self.sigma for val in node) )
        else:
            return Node( *(1 / len(node) for n in node) )

    def buildMatricesFromObservations (self, emissionString, path):
        emissions = {state: { obs: np.longdouble(0.) for obs in self.emits} for state in self.states}
        A = {fm: { to: np.longdouble(0.) for to in self.states} for fm in self.states}

        emissions[path[0:1]][emissionString[0:1]] += 1.
        fm = path[0:1]
        for obs, state in zip(emissionString[1:], path[1:]):
            emissions[state][obs] += 1.
            A[fm][state] += 1.
            fm = state

        self.emission = {state: self._normalize(emissions[state]) for state in self.states}
        self.transition = {fm: self._normalize(A[fm]) for fm in self.states}

    def printTransition(self):
        v = '\t'.join(''.join([str(i) for i in s]) for s in self.states)
        print ('\t{}'.format(v))

        for s in self.states:
            v = ''.join([str(i) for i in s])
            print (v, end ='')
            for t in self.states:
                if self.transition[s][t] == 0.:
                    print('\t{0:d}'.format( 0 ), end='')
                else:
                    print('\t{0:<.3}'.format(self.transition[s][t]), end='')
            print ()


    def printEmission(self):
        print ('\t', end='')
        print ('\t'.join(self.emits))

        for s in self.states:
            v = ''.join([str(i) for i in s])
            print(v, end='')
            for e in self.emits:
                if self.emission[s][e] == 0. :
                    print('\t{0:d}'.format( 0 ), end='')
                else:
                    print('\t{0:<.3}'.format(self.emission[s][e]), end = '')
            print ()

class ViterbiGraph:
    def __init__(self, size, seq, transitions, emissions): # size is specified in number of viterbiCells
        self.start = ViterbiStart(transitions, emissions)
        last = self.start
        # set up the left hand column of deletes
        for i in range(size):
            last = ViterbiRow(transitions, emissions, up=last, rank = i+1)

        # set up the top insert row
        last = self.start
        trans = transitions['S'][('i', 0)]
        for c in seq:
            last = ViterbiColumn(transitions, emissions, rank=trans, char=c, left=last)
            trans = transitions[('i', 0)][('i', 0)] # for every time other than start to i0

            # fill in the remainder of this column
            prevUp = last # set to the top of this column
            prevLeft = prevUp.left.down

            for i in range(size):
                prevUp = ViterbiCell(transitions, emissions, rank = i+1, char = c, left=prevLeft, up = prevUp)
                prevLeft = prevLeft.down
        # set up end state. It is the last Cell of the graph build
        self.end = ViterbiEnd(transitions, emissions, left = prevUp)

    def traceback(self):
        backPath = []

        navi = {'m': lambda t: t.up.left, # match - these are used to find the prior cell
                'd': lambda t: t.up,      # delete
                'i': lambda t: t.left}    # insert

        p = self.end.left                 # every state in every cell retains the prior arg path that keys the max
        thisPrior = self.end.prior['d']   # the 'd' state in End is a matter of convenience
        while p is not self.start:        # p refers to the current cell
            backPath.append(thisPrior + str(p.rank))
            nextp = navi[thisPrior](p)
            thisPrior = p.prior[thisPrior]
            p = nextp

        backPath.reverse()
        return ' '.join(backPath)


    def __repr__(self):
        output = []
        row = self.start
        while row:
            column = row
            for node in 'mdi':
                line = []
                while column:
                    line.append('{0:<.1}{1}'.format(column.state[node], column.prior[node]) )
                    column = column.right
                column = row
                output.append('\t'.join(line))
            output.append('')
            row = row.down
        output.append ('End = {0:<.1}{1}'.format(self.end.state['d'], self.end.prior['d']))
        return ('\n'.join(output))


class ViterbiCell:
    def __init__(self, transitions, emissions, rank = None, char=None, left=None, up=None, right=None, down = None):
        self.state = {key:float('-inf') for key in 'mdi'}
        self.prior = {key: None for key in 'mdi'}
        self.left = left
        self.right= right
        self.up = up
        self.down = down
        # set up reverse links back to this cell
        if left:
            left.right = self
        if up:
            up.down = self

        self.transitions = transitions
        self.emissions = emissions
        self.rank = rank
        self.char = char

        self.update()

    def update(self):
        prevRank = self.rank - 1

        mTrans = {j: self.transitions[(j, prevRank)][('m',self.rank)] for j in 'mdi'}
        dTrans = {j: self.transitions[(j, prevRank)][('d', self.rank)] for j in 'mdi'}
        iTrans = {j: self.transitions[(j, self.rank)]    [('i', self.rank)] for j in 'mdi'}

        mEmit = self.emissions[('m', self.rank)][self.char]
        iEmit = self.emissions[('i', self.rank)][self.char]

        self.prior['m'], self.state['m'] = self.maxProd(self.up.left, mTrans)
        self.prior['d'], self.state['d'] = self.maxProd(self.up, dTrans)
        self.prior['i'], self.state['i'] = self.maxProd(self.left, iTrans)

        self.state['m'] *= mEmit
        self.state['i'] *= iEmit

    def maxProd(self, prev, d):
        k = sorted([(k,prev.state[k] * d[k]) for k in d.keys()], key = lambda a:a[1], reverse=True)

        return k[0]

class ViterbiStart(ViterbiCell):
    def update(self):
        self.prior['m'], self.state['m'] = ('S', 1.0)
        self.prior['d'], self.state['d'] = ('S', 1.0)
        self.prior['i'], self.state['i'] = ('S', 1.0)

class ViterbiEnd(ViterbiCell):
    def update(self):
        self.prior['m'], self.state['m'] = ('E', '')
        self.prior['d'], self.state['d'] = self.maxProd(self.left, {j: self.transitions[(j, self.left.rank)]['E'] for j in 'mdi'})
        self.prior['i'], self.state['i'] = ('E', '')

class ViterbiRow(ViterbiCell):
    def update(self):
        if self.rank is 1:
            self.prior['d'], self.state['d'] = ('d', self.up.state['d'] * self.transitions['S'][('d', 1)])
        else:
            prevRank = self.rank - 1
            self.prior['d'], self.state['d'] = ('d', self.up.state['d'] * self.transitions[('d', prevRank)] [('d', self.rank)])

class ViterbiColumn(ViterbiCell):
    def update(self):
        # self.rank has been used to handle the transition prob
        self.prior['i'], self.state['i'] = ('i', self.left.state['i'] * self.rank * self.emissions[('i', 0)][self.char])

def main ():
    iterations = int(sys.stdin.readline().rstrip())
    junk = sys.stdin.readline().rstrip()  # skip -----
    sequence = sys.stdin.readline().rstrip()
    junk = sys.stdin.readline().rstrip()  # skip -----
    alphabet = sys.stdin.readline().rstrip().split()
    junk = sys.stdin.readline().rstrip()  # skip -----
    states = sys.stdin.readline().rstrip().split()
    junk = sys.stdin.readline().rstrip()  # skip -----

    thisHMM = HMM(emits=alphabet, states = states)
    thisHMM.buildTransition()   # read the initial Transition matrix
    thisHMM.buildEmission()     # read the initial Emission matrix

    thisHMM.printTransition()
    thisHMM.printEmission()

    for i in range(iterations):
        # do the E and M steps of Baum Welch learning
        thisHMM.maximizePi(sequence)

    thisHMM.printTransition()
    print (junk)
    thisHMM.printEmission()

if __name__ == "__main__":
    main()
