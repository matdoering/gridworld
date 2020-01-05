from Actions import Actions
import copy

class StateGenerator:

    def generateState(self, gridWorld, actionType, oldActorCell):
        # performance optimization for new states to be considered
        # generate all possible states
        # by applying all agent actions
        # and environment effects on these actions

        possibleStates = []
        moveState = gridWorld.proposeMove(actionType)
        if moveState is None:
            # cant move
            moveState = oldActorCell
        possibleStates.append(moveState)
        if moveState != oldActorCell:
            # move may fail
            possibleStates.append(oldActorCell)
        return possibleStates
