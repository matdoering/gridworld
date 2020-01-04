# track belief in states

import numpy as np
from Actions import Actions
from GameLogic import GameLogic
from PolicyConfig import getDefaultPolicyConfig
import random
from Action import Action
from Perception import Perception

def interpretBelief(P, gridWorld):
    i = np.argmax(P)
    cell = gridWorld.getCellByIndex(i)
    maxP = P[i]
    maxP_s = str(round(maxP*100, 2))+ "%"
    print("Max Belief is: " + maxP_s + ", at: " + str(cell.getCoords()))

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

    def exploreRandomly(self):
        # let agent randomly explore and update the belief over the state
        # start in random cell
        curCell = self.gridWorld.getRandomEnterableCell()
        self.gridWorld.setActor(curCell)
        print(self.gridWorld)
        actions = [a for a in Actions if a != Actions.NONE]
        curBelief = self.uniformPriorOverReachableStates()

        for i in range(500):
            # randomly pick an action
            a = random.choice(actions)
            #a = Actions.GO_NORTH
            #print(a)
            p = self.gridWorld.apply(a)
            #print(p)
            #curCell = self.gridWorld.getActorCell()
            curBelief = self.bayesFilter(Action(a), curBelief)
            curBelief = self.bayesFilter(Perception(p), curBelief)
            interpretBelief(curBelief, self.gridWorld)
            print(curBelief)
            print(self.gridWorld)


# TODO: implement QMDP algorithm (Q learning, belief convergence)
# for t-step policy, the value of executing an action in state s is:
# Vp(s) = R(s, a(p)) + expected reward in the future
# Vp(b) = sum_s [ belief(s)*  V_p(s)
