from Map import Map

import numpy as np
from Policy import Policy, improvePolicy, policyIteration # , valueIteration
from MapParser import MapParser
from PolicyParser import PolicyParser
from PolicyConfig import PolicyConfig, StickyWallConfig
from Belief import Belief, randomActionSelection, QMDP
from Actions import Actions
from Action import Action
from Perception import Perception
from Perceptions import Perceptions

def loadDefaultMap():
    parser = MapParser()
    gridMap = parser.parseMap("../data/map01.grid")
    return gridMap

def loadDefaultPolicy():
    parser = PolicyParser()
    policy = parser.parsePolicy("../data/map01.policy")
    return policy

# load map
gridMap = loadDefaultMap()
print(gridMap)

# options:
beliefTracking = True
testStickyWall = False
toImprovePolicy = False
selectPolicy = False

# run:
optimalPolicyThroughImprovement = None
optimalPolicyThroughValueIteration = None

if beliefTracking:
    bel = Belief(gridMap)
    #bel.explore(randomActionSelection)
    bel.explore(QMDP)

if testStickyWall:
    config = PolicyConfig(setStickyWalls = True)
    emptyPolicy = Policy([])
    emptyPolicy.setConfig(config)
    emptyPolicy.valueIteration(gridMap)
    print(emptyPolicy)
    emptyPolicy.resetValues()
    print("sticky policy:")
    print(emptyPolicy)
    for pSticky in [0.25,0.5,0.75, 0.9]:
        print("pSticky: " + str(pSticky))
        config.setStickyWallConfig(StickyWallConfig(pSticky))
        emptyPolicy.valueIteration(gridMap)
        emptyPolicy.resetValues()
        print(emptyPolicy)

if toImprovePolicy:
    # load default policy
    policy = loadDefaultPolicy()
    print(policy)
    # evaluate policy
    print("eval policy")
    V = policy.evaluatePolicy(gridMap)
    print(policy)
    print("improve policy")
    greedyPolicy = improvePolicy(policy, gridMap)
    print(greedyPolicy)
    greedyPolicy.resetValues()
    print(greedyPolicy)
    # policy iteration
    print("policy iteration:")
    optimalPolicyThroughImprovement = policyIteration(greedyPolicy, gridMap)
    print(optimalPolicyThroughImprovement)

if selectPolicy:
    # value iteration
    emptyPolicy = Policy([])
    print("value iteration")
    V_opt = emptyPolicy.valueIteration(gridMap)
    print(emptyPolicy)
    emptyPolicy.resetValues()
    print("optimal, non-sticky policy")
    print(emptyPolicy)
    optimalPolicyThroughValueIteration = emptyPolicy

resultOK = (optimalPolicyThroughImprovement == optimalPolicyThroughValueIteration)
print("both optimal: " + str(resultOK))
