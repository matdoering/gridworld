from Perception import Perception
from Perceptions import Perceptions
from Actions import Actions

def getPerceptionProbabilityWallDir(newCell):
    return 1 if newCell.isWall() else 0

class GameLogic:

    def __init__(self, config):
        self.config = config

    def getTransitionProbability(self, oldState, newState, action, gridWorld):
        # check whether move is possible at all
        proposedCell = gridWorld.proposeMove(action)
        if proposedCell is None:
            # action didn't have an effect (e.g. entered a wall)
            # -> definitely inquire penalty (= reward)
            return 1

        if gridWorld.isCellAdjacentToWall(oldState):
            stickyWallConfig = self.config.getStickyWallConfig()
            pStuck = stickyWallConfig.p # probability to be stuck next to wall
            if oldState == newState:
                # sticky wall had an effect
                return pStuck
            else:
                # sticky wall didn't have an effect with prob 1 - pStuck
                return 1 - pStuck
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

