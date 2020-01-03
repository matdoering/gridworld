# track belief in states

import numpy as np
from Actions import Actions
from GameLogic import GameLogic
from PolicyConfig import getDefaultPolicyConfig

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
        prior = self.uniformPriorOverReachableStates()
        print(prior)
        newBeliefs = np.zeros(len(belief))
        if dataItem.isPerception():
            eta = 0.0 # belief normalizer
            for cell in self.gridWorld.getCells():
                P = self.gameLogic.getPerceptionProbability(dataItem, cell)
                newBelief = P * belief[cell.getIndex()]
                newBeliefs[cell.getIndex()] = newBelief
                eta += newBelief
            # normalize belief
            newBeliefs /= eta
        elif dataItem.isAction():
            for newCell in self.gridWorld.getCells():
                newBelief = 0
                for oldCell in self.gridWorld.getCells():
                    # p(newCell | dataItem, oldCell)
                    P = self.gameLogic.getTransitionProbability(oldCell, newCell, data.getActionType(), self.gridWorld)
                    bel = belief[oldCell.getIndex()]
                    newBelief += P * bel
                newBeliefs[newCell.getIndex()] = newBelief

        else:
            raise Exception("Unhandled data item")



