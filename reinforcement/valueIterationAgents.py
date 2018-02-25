# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            new_vals = self.values.copy()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue

                actions = self.mdp.getPossibleActions(state)
                new_vals[state] = max([self.getQValue(state, a) for a in actions])

            self.values = new_vals


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q = 0
        for S_next, T in self.mdp.getTransitionStatesAndProbs(state, action):
            R = self.mdp.getReward(state, action, S_next)
            Q += T * (R + self.discount * self.values[S_next])

        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        policies = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            policies[action] = self.getQValue(state, action)

        return policies.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            new_vals = self.values.copy()
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]

            if self.mdp.isTerminal(state):
                continue
            # only update 1 state through each iteration 
            actions = self.mdp.getPossibleActions(state)
            new_vals[state] = max([self.getQValue(state, a) for a in actions])
            self.values = new_vals

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors, pq, states = {}, util.PriorityQueue(), self.mdp.getStates()

        # Get predecessors of all states
        for s in states:
            predecessors[s] = self.getPredecessors(s)

        # Add all non-terminal states to priority queue
        for s in states:
            if not self.mdp.isTerminal(s):
                diff = abs(self.values[s] - self.highestQValue(s))
                pq.update(s, -diff) # min heap priority queue

        # update q-values and priority queue
        for i in range(self.iterations):
            if pq.isEmpty():
                return

            s = pq.pop()
            self.values[s] = self.highestQValue(s)
            for p in list(predecessors[s]):
                if not self.mdp.isTerminal(p):
                    diff = abs(self.values[p] - self.highestQValue(p))
                    if diff > self.theta:
                        pq.update(p, -diff)  # min heap priority queue

    def highestQValue(self, state):
        """
          return the highest Q-value based on all possible actions of state
        """

        actions = self.mdp.getPossibleActions(state)
        Q = max([self.getQValue(state, a) for a in actions])

        return Q


    def getPredecessors(self, state):
        """
          return a set containing all predecessors of state
        """

        predecessorSet = set()

        if not self.mdp.isTerminal(state):
            for s in self.mdp.getStates():
                if not self.mdp.isTerminal(s):
                    for a in self.mdp.getPossibleActions(s):
                        for ns, t in self.mdp.getTransitionStatesAndProbs(s, a):
                            if (ns == state) and (t > 0):
                                predecessorSet.add(s)

        return predecessorSet