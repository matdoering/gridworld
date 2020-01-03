from Actions import Actions
import copy

class StateGenerator:

    def generateState(self, gridWorld, actionType, oldActorCell):
        # generate all possible states
        # by applying all agent actions
        # and environment effects on these actions

        possibleStates = []
        moveState = gridWorld.proposeMove(actionType)
        if moveState is None:
            # cant move
            moveState = oldActorCell
        possibleStates.append(moveState)

        if gridWorld.isCellAdjacentToWall(oldActorCell):
            # may need to consider transition failure in this case
            possibleStates.append(oldActorCell)
        return possibleStates
