from Map import Map

import numpy as np
from generated.proto import gridworld_pb2
from Policy import Policy, improvePolicy, policyIteration # , valueIteration
from MapParser import MapParser
from PolicyParser import PolicyParser
from PolicyConfig import PolicyConfig

import pickle

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
testStickyWall = True
toImprovePolicy = False
selectPolicy = False

# run:
optimalPolicyThroughImprovement = None
optimalPolicyThroughValueIteration = None

if testStickyWall:
    config = PolicyConfig(setStickyWalls = True)
    emptyPolicy = Policy([])
    emptyPolicy.setConfig(config)
    emptyPolicy.valueIteration(gridMap)
    print(emptyPolicy)
    emptyPolicy.resetValues()
    print("sticky policy:")
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
