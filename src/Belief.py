# track belief in states
import time
import numpy as np
from Actions import Actions, getViableActions
from GameLogic import GameLogic
from PolicyConfig import getDefaultPolicyConfig
import random
from Action import Action
from Perception import Perception
from StateGenerator import StateGenerator
from Policy import Policy

def interpretBelief(P, gridWorld):
    i = np.argmax(P)
    cell = gridWorld.getCellByIndex(i)
    maxP = P[i]
    maxP_s = str(round(maxP*100, 2))+ "%"
    print("Max Belief is: " + maxP_s + ", at: " + str(cell.getCoords()))

def randomActionSelection(gridWorld, belief, gameLogic, V):
        # ignores gridWorld and belief. arg just passed to have the same interface
        actions = getViableActions()
        a = random.choice(actions)
        return(a)


def QMDP(gridWorld, belief, gameLogic, V = None):
    # QMDP: approximation of PMDP using a hybrid of MDP and PMDP
    # value function: ignores state uncertainty
    # TODO: need +/-1 penalty for this to work. otherwise belief doesnt really have an effect
    policy = Policy([])
    if V is None:
        V = policy.valueIteration(gridWorld)
    generator = StateGenerator()
    actions = getViableActions()
    bestAction = (Actions.NONE, -np.inf) # pair of action and reward
    allBestActions = []
    #arrDim = (len(gridWorld.getCells()), len(gridWorld.getCells()), len(actions))
    #print(arrDim)
    #actionMap = np.zeros(arrDim)
    for action in actions:
        for cell in gridWorld.getViableCells():
            beliefInState = belief[cell.getIndex()]
            if beliefInState == 0:
                # hacky way to ignore cells without belief (would lead to max gain of 0 ...)
                #print("cell with belief 0 is: ", cell.getCoords())
                continue
            gridWorld.setActor(cell)
            newCell = gridWorld.proposeMove(action)
            immediateReward = policy.R(cell, newCell, action) # TODO: remove rewards from policy class and move into GameLogic
            possibleStates = generator.generateState(gridWorld, action, cell)
            expectedReward = 0.0
            for possibleState in possibleStates:
                P = gameLogic.getTransitionProbability(cell, possibleState, action, gridWorld)
                R = V[possibleState.getIndex()] * P
                expectedReward += R
            q = immediateReward + expectedReward
            #print(q)
            #if cell and cell.getRow() == 1 and cell.getCol() == 16:
                    #print(belief[cell.getIndex()], cell.getCoords(), action, P, R, immediateReward)
            # TODO: with current reward values, PQ will be highest (= 0) when belief = 0
            # TODO: alternative: dont force belief to 0 for unreachable cells
            PQ = beliefInState * q
            #actionMap[cell.getRow(), cell.getCol(), action.value-1] = PQ
            #print(PQ)
            if PQ > bestAction[1]:
                #print("new max at cell: ", cell.getCoords(), PQ)
                bestAction = (action, PQ)
                allBestActions = []
                allBestActions.append(action)
            elif PQ == bestAction[1]:
                #print("additional max at cell: ", cell.getCoords(), PQ)
                allBestActions.append(action)
    #print("QMDP choice: ", bestAction)
    #print(actionMap)
    print("No of optimal actions: " + str(len(allBestActions)))
    return random.choice(allBestActions)


class Belief:

    def __init__(self, gridWorld):
        self.gridWorld = gridWorld
        self.gameLogic = GameLogic(getDefaultPolicyConfig())

    def uniformPrior(self):
        # uniform prior over all states
        return np.repeat(1.0 / self.gridWorld.size(), self.gridWorld.size())

    def uniformPriorOverReachableStates(self):
        # uniform prior over reachable states
        prior = np.zeros(self.gridWorld.size())
        reachableStateIdx = []
        for cell in self.gridWorld.getCells():
            if cell.canBeEntered():
                reachableStateIdx.append(cell.getIndex())
        prior[reachableStateIdx] = 1.0/len(reachableStateIdx)
        return(prior)

    def bayesFilter(self, dataItem, belief):
        # alternatives: Kalmann filter, particle filter
        oldActorCell = self.gridWorld.getActorCell()
        newBeliefs = np.zeros(len(belief))
        if dataItem.isPerception():
            eta = 0.0 # belief normalizer
            for cell in self.gridWorld.getViableCells():
                self.gridWorld.setActor(cell) # necessary for proposeMove functionality at the moment (TODO?)
                P = self.gameLogic.getPerceptionProbability(dataItem, cell, self.gridWorld)
                self.gridWorld.unsetActor(cell)
                newBelief = P * belief[cell.getIndex()]
                newBeliefs[cell.getIndex()] = newBelief
                eta += newBelief
            # normalize belief
            if eta != 0.0:
                newBeliefs /= eta
        elif dataItem.isAction():
            for newCell in self.gridWorld.getViableCells():
                newBelief = 0
                for oldCell in self.gridWorld.getViableCells():
                    # p(newCell | dataItem, oldCell)
                    self.gridWorld.setActor(oldCell)
                    P = self.gameLogic.getTransitionProbability(oldCell, newCell, dataItem.getType(), self.gridWorld)
                    self.gridWorld.unsetActor(oldCell)
                    bel = belief[oldCell.getIndex()]
                    newBelief += P * bel
                newBeliefs[newCell.getIndex()] = newBelief
        else:
            raise Exception("Unhandled data item")
        self.gridWorld.setActor(oldActorCell) # reset actor
        return(newBeliefs)

    def explore(self, actionSelectionStrategy):
        # explore gridworld using 'actionSelectionStrategy' (either QMDP or random)

        # start in random cell
        curCell = self.gridWorld.getRandomEnterableCell()
        #curCell = self.gridWorld.getCell(1,16)
        self.gridWorld.setActor(curCell)
        print(self.gridWorld)
        actions = getViableActions()
        curBelief = self.uniformPriorOverReachableStates()

        policy = Policy([])
        V = policy.valueIteration(self.gridWorld)
        # TODO: implement python module that collects stats about agent operations
        # - number of actions taken
        # - shortest path from initial position
        # - actual actions taken / shortest path from initial
        while True:
            for i in range(100):
                # randomly pick an action
                a = actionSelectionStrategy(self.gridWorld, curBelief, self.gameLogic, V)
                self.gridWorld.setActor(curCell) # make sure that actor is on same position as before selecting action (TODO)
                #a = Actions.GO_NORTH
                #print(a)
                p = self.gridWorld.apply(a)
                curCell = self.gridWorld.getActorCell()
                #print(p)
                #curCell = self.gridWorld.getActorCell()
                curBelief = self.bayesFilter(Action(a), curBelief)
                curBelief = self.bayesFilter(Perception(p), curBelief)
                interpretBelief(curBelief, self.gridWorld)
                #print(curBelief)
                print(self.gridWorld)
                # check whether agent has reached the goal
                if self.gridWorld.getActorCell().isGoal():
                    print("#" * 30)
                    print("Yay, I reached the goal in iteration: " + str(i))
                    print("#" * 30)
                    time.sleep(1.5)
                    # reset to initial state:
                    curBelief = self.uniformPriorOverReachableStates()
                    self.gridWorld.unsetActor(curCell)
                    curCell = self.gridWorld.getRandomEnterableCell()
                    break
        return(i)

# problem: belief state is too unsure about the world to derive viable actions

# for t-step policy, the value of executing an action in state s is:
# Vp(s) = R(s, a(p)) + expected reward in the future
# Vp(b) = sum_s [ belief(s)*  V_p(s)
