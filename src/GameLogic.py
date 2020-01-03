
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

