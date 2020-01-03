from Map import Map
import copy
from enum import Enum
#from MapParser import ParseAction, symbolToEnum
from Actions import Actions
from Policy import Policy
from Action import Action

def actionCharToAction(actionChar):
    goNorth = Action(Actions.GO_NORTH)
    goEast = Action(Actions.GO_EAST)
    goSouth = Action(Actions.GO_SOUTH)
    goWest = Action(Actions.GO_WEST)
    mapping =  {
        str(goNorth): goNorth,
        str(goEast): goEast,
        str(goSouth): goSouth,
        str(goWest): goWest
    }
    action = None
    try:
        action = mapping[actionChar]
    except(Exception):
        action = Action(Actions.NONE)
    return action

class PolicyParser:

    def parseLine(self, line):
        actions = []
        for a in line:
            if a == "\n":
                break
            parseAction = actionCharToAction(a)
            actions.append(parseAction)
        return actions

    def parsePolicy(self, fname):
        width = None
        height = 0
        policy = []
        with open(fname) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                if width != None and (len(line) -1) != width:
                    raise Exception("Input width inconsistent")
                width = len(line) - 1
                height += 1
                rowActions = self.parseLine(line)
                policy.extend(rowActions)
        policy = Policy(policy)
        policy.setWidth(width)
        policy.setHeight(height)
        return(policy)
