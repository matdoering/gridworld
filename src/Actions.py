from enum import Enum

class Actions(Enum):
    GO_NORTH = 1
    GO_EAST = 2
    GO_SOUTH = 3
    GO_WEST = 4
    NONE = 5

def getViableActions():
    return [a for a in Actions if a != Actions.NONE]

