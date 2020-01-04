#from generated.proto import gridworld_pb2
from Actions import Actions
import sys
import random
from Perceptions import Perceptions

class Map:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells = []
        self.actorCell = None
        self.neighborCells = None

    def __eq__(self, other):
        return self.cells == other.cells and self.actorCell == other.actorCell

    def getActorCell(self):
        return self.actorCell

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def size(self):
        return self.width * self.height

    def setWidth(self, width):
        self.width = width

    def setHeight(self, height):
        self.height = height

    def getViableCells(self):
        return [c for c in self.cells if c.canBeEntered()]

    def getCells(self):
        return self.cells

    def setCells(self, cells):
        self.cells = cells
        self.setCellNeighbors()

    def getRandomEnterableCell(self):
        okCells = self.getViableCells()
        startCell = random.choice(okCells)
        return(startCell)

    def setCellNeighbors(self):
        self.neighborCells = []
        for cell in self.cells:
            #cell.printCoords()
            neighbors = []
            for otherCell in self.cells:
                # compute L1 distance (Manhattan)
                dist = abs(otherCell.getRow() - cell.getRow()) +\
                        abs(otherCell.getCol() - cell.getCol())
                if dist == 1:
                    # direct neighbors
                    neighbors.append(otherCell)
            self.neighborCells.append(neighbors)

    def isCellAdjacentToWall(self, cell):
        neighbors = self.neighborCells[cell.getIndex()]
        for neighborCell in neighbors:
            if neighborCell.isWall():
                return True
        return False

    def getCellNeighbors(self, cell):
        return self.neighborCells[cell.getIndex()]

    def getWallNeighbors(self, cell):
        neighbors = self.neighborCells[cell.getIndex()]
        wallNeighbors = [nCell for nCell in neighbors if nCell.isWall()]
        return(wallNeighbors)

    def getNbrAdjacentWalls(self, cell):
        return(len(self.getWallNeighbors(cell)))

    def getCellByIndex(self, idx):
        return self.cells[idx]

    def getCell(self, row, col):
        idx = (row * self.getWidth()) + col
        #print("getCell @ " + str(idx) + ", row: " + str(row) + ", col: " + str(col))
        return self.cells[idx]

    def getRow(self, row):
        rowStart = self.getWidth() * row
        rowEnd = rowStart + self.getWidth()
        #print("row int: " + str(rowStart) + ", " + str(rowEnd))
        row = self.cells[rowStart:rowEnd]
        return row

    def __str__(self):
        out = ""
        for rowId in range(self.getHeight()):
            rowCells = self.getRow(rowId)
            for cell in rowCells:
                out += str(cell)
            # row has ended
            out += "\n"
        return(out)

    def moveActor(self, newCell):
        self.getActorCell().unsetActor()
        self.actorCell = newCell
        newCell.setActor()

    def apply(self, action):
        # applies action to map
        return self.applyMove(action)

    def evaluateAction(self, action, actorCell):
        x = actorCell.getRow()
        y = actorCell.getCol()
        try:
            if action == Actions.GO_NORTH:
                potentialCell = self.getCell(x-1, y)
            elif action == Actions.GO_EAST:
                potentialCell = self.getCell(x, y+1)
            elif action == Actions.GO_SOUTH:
                potentialCell = self.getCell(x+1, y)
            elif action == Actions.GO_WEST:
                potentialCell = self.getCell(x, y-1)
            elif action == Actions.NONE:
                potentialCell = actorCell
            else:
               raise Exception("Unknown action")
        except(IndexError):
            # we're at the border of the map
            # so action can't be executed
            potentialCell = None
        return(potentialCell)

    def proposeMove(self, action):
        # movement logic
        #print("apply move:" + str(action))
        potentialCell = None
        actorCell = self.getActorCell()
        if actorCell.isGoal():
            # do not move away from goal
            return potentialCell
        if not actorCell.canBeEntered():
            # current actor cell is invalid (e.g. wall)
            #raise(Exception, "Actor in invalid cell") # TODO: implement exception to disallow this state
            return potentialCell
        potentialCell = self.evaluateAction(action, actorCell)
        if not potentialCell.canBeEntered():
            return None
        return potentialCell

    def getSensorReadOut(self, action, actorCell, useSmartBumpers = True):
        potentialCell = self.evaluateAction(action, actorCell)
        perception = None
        # TODO: put smart bumper use in GameConfig
        # TODO: move sensor read out to game logic
        if potentialCell and potentialCell.isWall():
            perception = Perceptions.HIT_WALL
            if useSmartBumpers:
                if action == Actions.GO_NORTH:
                    perception = Perceptions.HIT_WALL_N
                elif action == Actions.GO_EAST:
                    perception = Perceptions.HIT_WALL_E
                elif action == Actions.GO_SOUTH:
                    perception = Perceptions.HIT_WALL_S
                elif action == Actions.GO_EAST:
                    perception = Perceptions.HIT_WALL_E
        else:
            perception = Perceptions.NOT_HIT_WALL
        return perception

    def applyMove(self, action):
        oldActorCell = self.getActorCell()
        potentialCell = self.proposeMove(action)
        if potentialCell:
            # move actor
            self.moveActor(potentialCell)
        # return sensor measurements resulting from move
        return self.getSensorReadOut(action, oldActorCell)

    def setActor(self, cell):
        if self.actorCell:
            self.actorCell.unsetActor()
        self.actorCell = cell
        if cell:
            cell.setActor()

    def unsetActor(self, cell):
        self.actorCell = None
        cell.unsetActor()

    def hasActorReachedGoal(self):
        return self.actorCell.isGoal()

    #def serialize(self):
        #state = gridworld_pb2.State()
        #state.width = self.width
        #state.height = self.height


