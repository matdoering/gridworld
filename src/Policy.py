from Actions import Actions
import numpy as np
from StateGenerator import StateGenerator
import copy
import sys
from Action import Action
from PolicyConfig import PolicyConfig, getDefaultPolicyConfig
from GameLogic import GameLogic
import warnings
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

def medianValue(V):
    return np.median(V)

def storeValueFunctionInIter(gridWorld, policy, iterLabel):
    V = policy.getValues()
    medianV = medianValue(V)
    metaInfo = "Iteration: " + iterLabel + "\nMedian value: " + str(medianV)
    drawValueFunction(V, gridWorld, policy.getPolicy(), False, metaInfo)
    plt.savefig("../data/output/policy_iteration_" + iterLabel + ".png")
    plt.close()

def drawValueFunction(V, gridWorld, policy, showFigure = True, metaInfo = None):
    fig, ax = plt.subplots()
    fig.suptitle('Gridworld: Value Function and Policy', fontsize=18)
    plt.xlabel('Grid Col', fontsize=16)
    plt.ylabel('Grid Row', fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(left = 0.10, right = 0.95, bottom = 0.1, top = 0.85) # less white space
    im = ax.imshow(np.reshape(V, (-1, gridWorld.getWidth())))
    for cell in gridWorld.getCells():
        p = cell.getCoords()
        i = cell.getIndex()
        if not cell.isGoal():
            text = ax.text(p[1], p[0], str(policy[i]),
                       ha="center", va="center", color="w", size = 16)
        if cell.isGoal():
            text = ax.text(p[1], p[0], "X",
                       ha="center", va="center", color="w", size = 16)
    if metaInfo:
       #fig.text(1.0, 1.0, metaInfo, ha='right', va='top', transform=ax.transAxes)
       fig.text(0.6, 0.91, metaInfo, ha='left', va='top', size = 12)

    if showFigure:
        plt.show()
    return(fig)

def initValues(gridWorld):
    values = np.zeros(gridWorld.size())
    for cell in gridWorld.getCells():
        if not cell.canBeEntered():
            values[cell.getIndex()] = -np.inf
    return values

def findGreedyPolicy(values, gridWorld, gameLogic, gamma = 1):
    # create a greedy policy based on the values param
    stateGen = StateGenerator()
    greedyPolicy = [Action(Actions.NONE)] * len(values)
    for (i, cell) in enumerate(gridWorld.getCells()):
        gridWorld.setActor(cell)
        if not cell.canBeEntered():
            continue
        maxPair = (Actions.NONE, -np.inf)
        for actionType in Actions:
            if actionType == Actions.NONE:
                continue
            proposedCell = gridWorld.proposeMove(actionType)
            if proposedCell is None:
                # action is nonsensical in this state
                continue
            Q = 0.0 # action-value function
            proposedStates = stateGen.generateState(gridWorld, actionType, cell)
            for proposedState in proposedStates:
                actorPos = proposedState.getIndex()
                transitionProb = gameLogic.getTransitionProbability(cell, proposedState, actionType, gridWorld)
                reward = gameLogic.R(cell, proposedState, actionType)
                expectedValue = transitionProb * (reward + gamma * values[actorPos])
                Q += expectedValue
            if Q > maxPair[1]:
                maxPair = (actionType, Q)
        gridWorld.unsetActor(cell) # reset state
        greedyPolicy[i] = Action(maxPair[0])
    return greedyPolicy

def improvePolicy(policy, gridWorld, gamma = 1):
    policy = copy.deepcopy(policy) # dont modify old policy
    if len(policy.values) == 0:
        # policy needs to be evaluated first
        policy.evaluatePolicy(gridWorld, gamma)
    #print("new values:")
    #print(policy.getValues())
    greedyPolicy = findGreedyPolicy(policy.getValues(), gridWorld, policy.gameLogic)
    policy.setPolicy(greedyPolicy)
    return policy

def policyIteration(policy, gridWorld, gamma = 1, storeValueFunction = False):
    # iteratively improve policy by cycling
    # policy evaluation and policy improvement
    t = time.time()
    print("Input policy:")
    print(policy)
    if storeValueFunction:
        storeValueFunctionInIter(gridWorld, policy, "00")
    lastPolicy = copy.deepcopy(policy)
    lastPolicy.resetValues() # reset values to force re-evaluation of policy
    improvedPolicy = None
    it = 0
    while True:
        it += 1
        improvedPolicy = improvePolicy(lastPolicy, gridWorld, gamma)
        if storeValueFunction:
            label = str(it)
            if it < 10:
                label = "0" + str(it)
            storeValueFunctionInIter(gridWorld, improvedPolicy, label)
        #print("policyIteration: " + str(it))
        #print(lastPolicy)
        #print(improvedPolicy) # DEBUG
        if improvedPolicy == lastPolicy:
            break
        improvedPolicy.resetValues() # to force re-evaluation of values on next run
        lastPolicy = improvedPolicy
    t = time.time() - t
    print("Policy iteration terminated after: " + str(it) + " iterations, time: " + str(t))
    return(improvedPolicy)

class Policy:

    def __init__(self, policy):
        self.policy = policy
        self.width = None
        self.height = None
        self.values = np.zeros(0)
        self.gameLogic = GameLogic(getDefaultPolicyConfig())

    def getPolicy(self):
        return self.policy

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
        return self.policy[cell.getIndex()].getType()

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
        return self.gameLogic.getTransitionProbability(oldState, newState, actionType, gridWorld)

    def R(self, oldState, newState, action):
        return self.gameLogic.R(oldState, newState, action)

    def evaluatePolicy(self, gridWorld, gamma = 1):
        # determine the value function V using policy iteration
        # map: gridworld map
        # gamma: discount rate

        t = time.time()
        if len(self.policy) != len(gridWorld.getCells()):
            # sanity check whether policy matches dimension of gridWorld
            raise Exception("Policy dimension doesn't fit gridworld dimension.")
        maxIterations = 500
        V_old = None
        V_new = initValues(gridWorld)
        iter = 0
        convergedCellIndices = np.zeros(0) # cells where values don't change anymore
        while len(convergedCellIndices) != len(V_new):
            V_old = V_new
            iter += 1
            V_new = self.evaluatePolicySweep(gridWorld, V_old, gamma, convergedCellIndices)
            convergedCellIndices = self.findConvergedCells(V_old, V_new)
            if iter > maxIterations:
                print("Terminated policy evaluation after " + str(maxIterations) + " iterations")
                break
        t = time.time() - t
        print("Policy evaluation converged after iteration: " + str(iter) + ", time: " + str(t))
        #print(V_new)
        return V_new

    def findConvergedCells(self, V_old, V_new, theta = 0.01):
        # returns list of cells where values haven't changed
        # optimization for policy evaluation such that known values aren't recomputed again

        # silence warnings from '-inf' values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered')
            diff = abs(V_old-V_new)
            idx = np.where(diff < theta)[0]
            sameIdx = np.where(V_new == -np.inf)[0]
            return np.concatenate((idx, sameIdx))

    def evaluatePolicySweep(self, gridWorld, V_old, gamma, ignoreCellIndices):
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
        stateGen = StateGenerator()
        transitionRewards = [-np.inf] * len(Actions)
        # perform full backup operation for this state
        for (i, actionType) in enumerate(Actions):
            gridWorld.setActor(cell) # set state
            actionProb = self.pi(cell, actionType)
            if actionProb == 0 or actionType == Actions.NONE:
                continue
            newStates = stateGen.generateState(gridWorld, actionType, cell)
            transitionReward = 0
            for newActorCell in newStates:
                V_newState = V_old[newActorCell.getIndex()]
                # Bellman equation
                newStateReward = self.P(cell, newActorCell, actionType, gridWorld) *\
                                    (self.R(cell, newActorCell, actionType) +\
                                    gamma * V_newState)
                transitionReward += newStateReward
            transitionRewards[i] = transitionReward
            V_a = actionProb * transitionReward
            V += V_a
        if len(self.policy) == 0:
            V = max(transitionRewards)
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
            V_new = self.evaluatePolicySweep(gridWorld, V_old, gamma, ignoreCellIndices)
            ignoreCellIndices = self.findConvergedCells(V_old, V_new)
        print("Terminated after: " + str(iter) + " iterations")
        # store policy found through value iteration
        greedyPolicy = findGreedyPolicy(V_new, gridWorld, self.gameLogic)
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
