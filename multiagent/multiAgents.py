# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        if action == "Stop":
            return -999999
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        for i, ghost in enumerate(newGhostStates):
            distToGhost = util.manhattanDistance(newPos, ghost.getPosition())
            safeDist = max(distToGhost, 0.1)

            if newScaredTimes[i] > 0:
                score += 100.0 / safeDist
            else:
                if distToGhost < 2:
                    return -999999
                else:
                    score -= 2.0 / safeDist

        foodList = newFood.asList()
        if foodList:
            foodDist = [util.manhattanDistance(newPos, food) for food in foodList]
            minFoodDist = min(foodDist)
            score -= 1.5 * minFoodDist
            score -= 10 * len(foodList)
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def value(state, depth, agentIndex):

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state), ""

            if agentIndex == 0:
                return maxValue(state, depth)

            else:
                return minValue(state, depth, agentIndex)

        def maxValue(state, depth):
            v = float("-inf")
            bestAction = ""
            actions = state.getLegalActions(0)

            for action in actions:

                successor = state.generateSuccessor(0, action)
                newV, _ = value(successor, depth, 1)
                if newV > v:
                    v, bestAction = newV, action

            return v, bestAction

        def minValue(state, depth, agentIndex):
            v = float("inf")
            bestAction = ""
            actions = state.getLegalActions(agentIndex)
            numAgents = state.getNumAgents()

            for action in actions:

                successor = state.generateSuccessor(agentIndex, action)

                if agentIndex == numAgents - 1:

                    newV, _ = value(successor, depth + 1, 0)
                else:

                    newV, _ = value(successor, depth, agentIndex + 1)

                if newV < v:
                    v, bestAction = newV, action

            return v, bestAction

        _, action = value(gameState, 0, 0)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.alphaBeta(
            gameState, self.depth, 0, float("-inf"), float("inf")
        )
        return action

    def alphaBeta(
        self,
        gameState: GameState,
        depth: int,
        agentIndex: int,
        alpha: float,
        beta: float,
    ):
        """
        Minimax with alpha-beta pruning.
        Returns (value, action) tuple.

        agentIndex: 0 = Pacman (maximizer), 1+ = Ghosts (minimizers)
        alpha: best value for maximizer
        beta: best value for minimizer
        """
        # Base case: terminal state or max depth reached
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        legalActions = gameState.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        # Decrement depth only after all agents move (when returning to Pacman)
        nextDepth = depth if nextAgent != 0 else depth - 1

        if agentIndex == 0:  # Pacman is maximizer
            maxValue = float("-inf")
            bestAction = None
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value, _ = self.alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
                # Update alpha
                if value > alpha:
                    alpha = value
                # Prune ONLY when maxValue > beta (strict inequality, not >=)
                if maxValue > beta:
                    break
            return maxValue, bestAction
        else:  # Ghost is minimizer
            minValue = float("inf")
            bestAction = None
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value, _ = self.alphaBeta(successor, nextDepth, nextAgent, alpha, beta)
                if value < minValue:
                    minValue = value
                    bestAction = action
                # Update beta
                if value < beta:
                    beta = value
                # Prune ONLY when minValue < alpha (strict inequality, not <=)
                if minValue < alpha:
                    break
            return minValue, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax(state, depth, agentIndex):
            """
            Recursive expectimax function.
            - agentIndex == 0  → Pacman (MAX node)
            - agentIndex >= 1  → Ghost  (CHANCE / expectation node)
            depth counts full plies; decremented only when we wrap back to Pacman (agent 0).
            """
            # Terminal test: game over OR reached depth limit
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents   = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            # Determine next agent and whether depth should decrease
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            successors = [
                state.generateSuccessor(agentIndex, action)
                for action in legalActions
            ]

            if agentIndex == 0:
                # MAX node – Pacman picks the best move
                return max(
                    expectimax(succ, nextDepth, nextAgent)
                    for succ in successors
                )
            else:
                # CHANCE node – ghost moves uniformly at random
                return sum(
                    expectimax(succ, nextDepth, nextAgent)
                    for succ in successors
                ) / len(successors)

        # Root call: choose the action that maximises the expectimax value
        legalActions = gameState.getLegalActions(0)
        bestAction   = max(
            legalActions,
            key=lambda action: expectimax(
                gameState.generateSuccessor(0, action),
                self.depth,
                1   # first ghost index
            )
        )
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance

    # Hard wins / losses
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pos         = currentGameState.getPacmanPosition()
    foodList    = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules    = currentGameState.getCapsules()
    score       = currentGameState.getScore()

    # ── Food ────────────────────────────────────────────────
    if foodList:
        minFoodDist = min(manhattanDistance(pos, f) for f in foodList)
        score += 10.0 / minFoodDist          # closer food → higher score
    score -= 4.0 * len(foodList)             # fewer pellets remaining is better

    # ── Ghosts ──────────────────────────────────────────────
    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Ghost is scared – chase it
            score += 200.0 / (dist + 1)
        else:
            # Ghost is dangerous
            if dist <= 1:
                score -= 500          # immediate danger
            else:
                score -= 2.0 / dist   # mild repulsion

    # ── Capsules ────────────────────────────────────────────
    score -= 20.0 * len(capsules)     # prefer states where capsules are eaten

    return score


# Abbreviation
better = betterEvaluationFunction
