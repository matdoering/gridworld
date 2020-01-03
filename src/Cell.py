from abc import ABC, abstractmethod
#from generated.proto import gridworld_pb2
import copy

# interface for cells
class Cell(ABC):

    @abstractmethod
    def canBeEntered(self):
        pass

    def __init__(self):
        self.row = 0
        self.col = 0
        self.containsGoal = False
        self.containsActor = False
        self.neighbors = []

    def printCoords(self):
        print(self.row, self.col)

    def addNeighborCell(self, cell):
        self.neighbors.append(cell)

    def getNeighborCells(self):
        return self.neighbors

    def setRow(self, row):
        self.row = row

    def setCol(self, col):
        self.col = col

    def setIndex(self, index):
        # index in cell array
        self.index = index

    def getIndex(self):
        return self.index

    def getRow(self):
        return self.row

    def getCol(self):
        return self.col

    def isGoal(self):
        return self.containsGoal

    def setActor(self):
        self.containsActor = True

    def unsetActor(self):
        self.containsActor = False

    def hasActor(self):
        return self.containsActor

    def __eq__(self, other):
        return(self.row == other.row and self.col == other.col and self.containsActor == other.containsActor)

    def __str__(self):
        return "@"

    def isWall(self):
        return False

    #def serialize(self):
        #cell = gridworld_pb2.Cell()
        #print(cell.SerializeToString())
