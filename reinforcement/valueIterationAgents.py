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

        for i in range(self.iterations):
            tempValues = util.Counter()
                
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                    
                bestAction = self.computeActionFromValues(state)
                    
                if bestAction is not None:
                    tempValues[state] = self.computeQValueFromValues(state, bestAction)
                
            self.values = tempValues
      

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
        qValue = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextState, prob in transitions:
            # R(s, a, s')
            reward = self.mdp.getReward(state, action, nextState)
            # V(s') 
            nextValue = self.getValue(nextState)
            
            qValue += prob * (reward + self.discount * nextValue)
            
        return qValue

        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
            
        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) == 0:
            return None
            
        bestAction = None
        maxQValue = float('-inf')
        
        for action in possibleActions:
            qValue = self.computeQValueFromValues(state, action)
            
            if qValue > maxQValue:
                maxQValue = qValue
                bestAction = action
                
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        
      
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            if nextState not in predecessors:
                                predecessors[nextState] = set()
                            predecessors[nextState].add(state)
        
      
        pq = util.PriorityQueue()

        
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
              
                best_action = self.computeActionFromValues(state)
                max_q_value = self.computeQValueFromValues(state, best_action)
                
               
                diff = abs(self.values[state] - max_q_value)
                pq.push(state, -diff)

       
        for iteration in range(self.iterations):
            if pq.isEmpty():
                break
            
            
            state = pq.pop()
            
          
            if not self.mdp.isTerminal(state):
                best_action = self.computeActionFromValues(state)
                max_q_value = self.computeQValueFromValues(state, best_action)
                self.values[state] = max_q_value
                
            
            for p in predecessors.get(state, []):
                if not self.mdp.isTerminal(p):
                   
                    best_action_p = self.computeActionFromValues(p)
                    max_q_value_p = self.computeQValueFromValues(p, best_action_p)
                    
                    diff_p = abs(self.values[p] - max_q_value_p)
                    
                
                    if diff_p > self.theta:
                        pq.update(p, -diff_p)

