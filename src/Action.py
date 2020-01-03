from Actions import Actions

class Action:
    def __init__(self, actionType):
        self.actionType = actionType

    def __eq__(self, other):
        return self.actionType == other.actionType

    def __str__(self):
        if self.actionType == Actions.GO_NORTH:
            return "N"
        elif self.actionType == Actions.GO_EAST:
            return "E"
        elif self.actionType == Actions.GO_SOUTH:
            return "S"
        elif self.actionType == Actions.GO_WEST:
            return "W"
        else:
            return "?"

    def getActionType(self):
        return self.actionType
