from Actions import Actions
import copy

class StateGenerator:

    def generateState(self, gridWorld, actionType):
        # generate all possible states
        # by applying all agent actions
        # and environment effects on these actions
        possibleStates = []
        oldActorCell = gridWorld.getActorCell()
        newState = gridWorld
        try:
            newState.apply(actionType)
        except(IndexError):
            # we're at the border of the map
            # so action can't be executed
            # -> return copy of previous state
            return [oldActorCell]
        possibleStates.append(newState.getActorCell())

        if gridWorld.isCellAdjacentToWall(oldActorCell):
            # need to consider transition failure in this case
            possibleStates.append(oldActorCell)
        newState.setActor(oldActorCell) # dont move actor for real
        return possibleStates
