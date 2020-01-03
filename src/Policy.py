from Actions import Actions
import numpy as np
from StateGenerator import StateGenerator
import copy
import sys
from Action import Action
from PolicyConfig import PolicyConfig, getDefaultPolicyConfig
from GameLogic import GameLogic

def initValues(gridWorld):
    values = np.zeros(gridWorld.size())
    for cell in gridWorld.getCells():
        if not cell.canBeEntered():
            values[cell.getIndex()] = -np.inf
    return values

def createPolicy(values, gridWorld):
    # create a greedy policy based on the values param
    #print("---------CREATE POLICY-----------")
    #print(values)
    workingState = copy.deepcopy(gridWorld)

    stateGen = StateGenerator()
    greedyPolicy = [Action(Actions.NONE)] * len(values)
    for (i, cell) in enumerate(workingState.getCells()):
        if not cell.canBeEntered():
            continue
        # select action maximizing value
        # TODO: hit
        #hit = False
        #if cell.getRow() == 1 and cell.getCol() == 16:
            #cell.printCoords()
            #hit = True

        maxPair = (Actions.NONE, -np.inf)
        for actionType in Actions:
            #if hit:
                #print(actionType)
            workingState.setActor(cell) # reset state
            if actionType == Actions.NONE:
                continue

            # exclude invalid moves (TODO)
            proposedState = workingState.proposeMove(actionType)
            if proposedState is None:
                continue

            newStates = stateGen.generateState(workingState, actionType)
            #if hit:
                #print(len(newStates))
            totalValue = 0.0
            for newActorCell in newStates:
                actorPos = newActorCell.getIndex()

                totalValue += values[actorPos]
                #if hit:
                #    print("hit coords:")
                #    newActorCell.printCoords()
            #if hit:
                #print(totalValue)
            if totalValue > maxPair[1]:
                #if hit:
                #    print("max value:")
                #    print(actionType)
                #    print(totalValue)
                maxPair = (actionType, totalValue)
        workingState.unsetActor(cell) # reset state
        #print(maxPair[0])
        greedyPolicy[i] = Action(maxPair[0])
    return greedyPolicy

def improvePolicy(policy, gridWorld):
    policy = copy.deepcopy(policy) # dont modify old policy
    if len(policy.values) == 0:
        # policy needs to be evaluated first
        policy.evaluatePolicy(gridWorld)
    #print("new values:")
    #print(policy.getValues())
    greedyPolicy = createPolicy(policy.getValues(), gridWorld)
    policy.setPolicy(greedyPolicy)
    return policy

def policyIteration(policy, gridWorld):
    # iteratively improve policy by cycling
    # policy evaluation and policy improvement
    print("Input policy:")
    print(policy)
    lastPolicy = copy.deepcopy(policy)
    lastPolicy.resetValues() # reset values to force re-evaluation of policy
    improvedPolicy = None
    it = 0
    while True:
        improvedPolicy = improvePolicy(lastPolicy, gridWorld)
        improvedPolicy.resetValues() # to force re-evaluation of values on next run
        it += 1
        #print("policyIteration: " + str(it))
        #print(lastPolicy)
        print(improvedPolicy) # DEBUG
        if improvedPolicy == lastPolicy:
            break
        lastPolicy = improvedPolicy
    return(improvedPolicy)

class Policy:

    def __init__(self, policy):
        self.policy = policy
        self.width = None
        self.height = None
        self.values = np.zeros(0)
        self.gameLogic = GameLogic(getDefaultPolicyConfig())

    def setConfig(self, policyConfig):
        self.gameLogic = GameLogic(policyConfig)

    def __eq__(self, other):
        if not isinstance(other, Policy):
            return False
        if self.width != other.width:
            return False
        if self.height != other.height:
            return False
        #for (p1, p2) in zip(self.policy, other.policy):
            #if p1 != p2:
                #print (p1, p2)
        return self.policy == other.policy

    def getValues(self):
        return self.values

    def setPolicy(self, policy):
        self.policy = policy

    def setValues(self, values):
        self.values = values

    def resetValues(self):
        self.values = np.zeros(0)

    def setWidth(self, width):
        self.width = width

    def setHeight(self, height):
        self.height = height

    def policyActionForCell(self, cell):
        #print("policyActionForCell")
        #print(self.policy[cell.getIndex()])
        #print(cell.getIndex())
        return self.policy[cell.getIndex()].getActionType()

    def pi(self, cell, action):
        # probability that policy performs action 'a' in state 's'
        if len(self.policy) == 0:
            # no policy: try all actions
            return 1

        if self.policyActionForCell(cell) == action:
            # policy allows this action
            return  1
        else:
            # policy forbids this action
            return 0

    def P(self, oldState, newState, actionType, gridWorld):
        # probability to transition from oldState to newState given action
        transitionProb = self.gameLogic.getTransitionProbability(oldState, newState, actionType, gridWorld)
        return(transitionProb)

    def R(self, oldState, newState, action):
        # reward for state transition from oldState to newState via action
        if newState.isGoal():
            return 0
        else:
            return - 1

    def evaluatePolicy(self, gridWorld, gamma = 1):
        # determine the value function V using policy iteration
        # map: gridworld map
        # gamma: discount rate

        if len(self.policy) != len(gridWorld.getCells()):
            # sanity check whether policy matches dimension of gridWorld
            raise Exception("Policy dimension doesn't fit gridworld dimension.")

        V_old = None
        V_new = initValues(gridWorld)
        iter = 0
        ignoreCellIndices = np.zeros(0) # cells where values don't change anymore
        while np.any(V_new != V_old):
            V_old = V_new
            iter += 1
            #print("iter: " + str(iter))
            V_new = self.evaluatePolicyIteration(gridWorld, V_old, gamma, ignoreCellIndices)
            #print(V_new)
            ignoreCellIndices = self.findConvergedCells(V_old, V_new)
        print("Policy evaluation terminated after iteration: " + str(iter))
        #print(V_new)
        return V_new

    def findConvergedCells(self, V_old, V_new, theta = 0.01):
        # returns list of cells where values haven't changed
        # optimization for policy evaluation such that known values aren't recomputed again
        idx = np.where(abs(V_old - V_new < theta))[0]
        return idx

    def evaluatePolicyIteration(self, gridWorld, V_old, gamma, ignoreCellIndices):
        V = initValues(gridWorld)
        # evaluate policy for every state (i.e. for every viable actor position)
        for (i,cell) in enumerate(gridWorld.getCells()):
            if np.any(ignoreCellIndices == i):
                V[i] = V_old[i]
            else:
                if cell.canBeEntered():
                    gridWorld.setActor(cell)
                    V_s = self.evaluatePolicyForState(gridWorld, V_old, gamma)
                    gridWorld.unsetActor(cell)
                    V[i] = V_s
        self.setValues(V)
        return V

    def evaluatePolicyForState(self, gridWorld, V_old, gamma):
        V = 0
        cell = gridWorld.getActorCell()
        #hit = False
        #if cell.getRow() == 1 and cell.getCol() == 16:
        #    cell.printCoords()
        #    hit = True
        #    print(V_old[cell.getIndex()])
        stateGen = StateGenerator()
        transitionRewards = [-np.inf] * len(Actions)
        # perform full backup operation for this state
        for (i, actionType) in enumerate(Actions):
            gridWorld.setActor(cell) # reset state
            actionProb = self.pi(cell, actionType)
            #if hit:
                #print(actionType)
                #print(gridWorld)
                #print("policy allowed action: " + str(actionProb))
            if actionProb == 0 or actionType == Actions.NONE:
                continue
            newStates = stateGen.generateState(gridWorld, actionType)
            #if hit:
                #print("no states: " + str(len(newStates)))
            transitionReward = 0
            for newActorCell in newStates:
                # TODO: simplify state handling here (setActor logic unncessary!? ...)
                gridWorld.setActor(cell) # reset state
                #if hit:
                #    print("new cell: " + str(newActorCell))
                V_newState = V_old[newActorCell.getIndex()]
                # Bellman equation performs bootstrapping:
                # estimate is updated using another estimate
                newStateReward = self.P(cell, newActorCell, actionType, gridWorld) *\
                                    (self.R(cell, newActorCell, actionType) +\
                                    gamma * V_newState)
                #if hit:
                    #print(newActorCell.printCoords())
                    #print("P:" + str(self.P(cell, newActorCell, actionType, gridWorld)))
                    #print("reward: " + str(newStateReward))

                transitionReward += newStateReward
                if transitionReward < -100:
                    print(transitionReward)
                    cell.printCoords()
                    sys.exit()

            transitionRewards[i] = transitionReward
            V_a = actionProb * transitionReward
            V += V_a
            #if hit:
                #print(transitionRewards)
        if len(self.policy) == 0:
            # value iteration
            #print(transitionRewards)
            #if hit:
                #print(transitionRewards)
            V = max(transitionRewards)
        #if hit:
            #print("---totalReward: " + str(V))
        return V

    def getValue(self, i):
        return self.values[i]

    def resetPolicy(self):
        self.policy = []

    def valueIteration(self, gridWorld, gamma = 1):
        # determine the value function V by combining
        # evaluation and policy improvement steps

        # reset policy to ensure that value iteration algorithm is used
        # instead of improving existing policy
        self.resetPolicy()

        V_old = None
        V_new = np.repeat(0, gridWorld.size())
        iter = 0
        ignoreCellIndices = np.zeros(0) # cells where values don't change anymore
        while np.any(V_new != V_old):
            V_old = V_new
            iter += 1
            V_new = self.evaluatePolicyIteration(gridWorld, V_old, gamma, ignoreCellIndices)
            #tempP= createPolicy(V_new, gridWorld)
            #tempPolicy = Policy(tempP)
            #tempPolicy.setWidth(gridWorld.getWidth())
            #tempPolicy.setHeight(gridWorld.getHeight())
            #tempPolicy.setValues(V_new)
            #print(tempPolicy)
            ignoreCellIndices = self.findConvergedCells(V_old, V_new)
        print("Terminated after: " + str(iter) + " iterations")
        # store policy found through value iteration
        greedyPolicy = createPolicy(V_new, gridWorld)
        self.setPolicy(greedyPolicy)
        self.setWidth(gridWorld.getWidth())
        self.setHeight(gridWorld.getHeight())
        return(V_new)

    def __str__(self):
        out = ""
        for (i,a) in enumerate(self.policy):
            if (i % self.width) == 0:
                out += "\n"
            if len(self.values) == 0:
                out += str(a)
            else:
                val = str(round(self.values[i], 1))
                while len(val) < 4:
                    val += " "
                out += val
        return(out)
