from Perception import Perception
from Perceptions import Perceptions
from Actions import Actions

def getPerceptionProbabilityWallDir(newCell):
    return 1 if newCell.isWall() else 0

def transitionProbabilityForIllegalMoves(oldState, newState):
    if newState == oldState:
        # this is correct
        return 1
    else:
        # cant transition to 'newState' if action is invalid
        return 0

def transitionProbabilityAdjacentToWall(oldState, newState, pStuck):
    # pStuck: probability to be stuck next to wall
    if oldState == newState:
        # sticky wall had an effect
        return pStuck
    else:
        # sticky wall didn't have an effect with prob 1 - pStuck
        return 1 - pStuck

class GameLogic:

    def __init__(self, config):
        self.config = config

    def getTransitionProbability(self, oldState, newState, action, gridWorld):
        # check whether move is possible at all
        proposedCell = gridWorld.proposeMove(action)
        if proposedCell is None:
            # illegal action for 'oldState'
            return transitionProbabilityForIllegalMoves(oldState, newState)
        if oldState != newState and newState != proposedCell:
            # newState cant be reached with available actions
            return 0
        if oldState.isGoal():
            if newState == oldState:
                # remain at goal
                return 0 # TODO
            else:
                # dont move away from goal cell
                return 0
        if gridWorld.isCellAdjacentToWall(oldState):
            stickyWallConfig = self.config.getStickyWallConfig()
            return transitionProbabilityAdjacentToWall(oldState, newState, stickyWallConfig.p)
        # normal game logic
        if proposedCell != newState:
            # state doesn't match expectation
            return 0
        else:
            # normal transition between states
            return 1


    def getPerceptionProbability(self, perception, cell, gridWorld):
        # probability of making 'perception' given current position ('cell')
        perceptionType = perception.getType()
        nbrNeighborCells = len(gridWorld.getCellNeighbors(cell))
        nbrAdjacentWalls = gridWorld.getNbrAdjacentWalls(cell)
        nbrAdjacentNonWalls = nbrNeighborCells - nbrAdjacentWalls
        nbrActions = 4 # TODO: determine programmatically using Actions class
        if perceptionType == Perceptions.HIT_WALL:
            # determine frequency of hitting wall when in state 'cell'
            return nbrAdjacentWalls / nbrActions
        elif perceptionType == Perceptions.HIT_WALL_N:
            newCell = gridWorld.evaluateAction(Actions.GO_NORTH, cell)
            return getPerceptionProbabilityWallDir(newCell)
        elif perceptionType == Perceptions.HIT_WALL_E:
            newCell = gridWorld.evaluateAction(Actions.GO_EAST, cell)
            return getPerceptionProbabilityWallDir(newCell)
        elif perceptionType == Perceptions.HIT_WALL_S:
            newCell = gridWorld.evaluateAction(Actions.GO_SOUTH, cell)
            return getPerceptionProbabilityWallDir(newCell)
        elif perceptionType == Perceptions.HIT_WALL_W:
            newCell = gridWorld.evaluateAction(Actions.GO_WEST, cell)
            return getPerceptionProbabilityWallDir(newCell)
        elif perceptionType == Perceptions.NOT_HIT_WALL:
            # determine frequency of not hitting wall when in state 'cell'
            return nbrAdjacentNonWalls / nbrActions
        else:
            raise(Exception, "Unhandled perception type")

    def R(self, oldState, newState, action):
        # reward for state transition from oldState to newState via action
        if newState and newState.isGoal():
            return 0
        else:
            return -1


