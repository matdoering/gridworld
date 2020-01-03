from Map import Map
from EmptyCell import EmptyCell
from GoalCell import GoalCell
from Wall import Wall
import copy
from enum import Enum

class ParseAction(Enum):
    NEW_LINE = 0
    WALL = 1
    EMPTY_CELL = 2
    START_CELL = 3
    GOAL_CELL = 4

def symbolToEnum(symbol):
    return {
            "#": ParseAction.WALL,
            " ": ParseAction.EMPTY_CELL,
            "O": ParseAction.START_CELL,
            "X": ParseAction.GOAL_CELL,
            "\n": ParseAction.NEW_LINE
    }[symbol]

def symbolToEntity(symbol):
       return copy.deepcopy({
            "#": Wall(),
            " ": EmptyCell(),
            "X": GoalCell()
    }[symbol])

def initializeCells(cells, width):
    for (i, cell) in enumerate(cells):
        # set x,y of cell
        colIdx = i % width
        cell.setCol(colIdx)
        rowIdx = int(i / width)
        cell.setRow(rowIdx)
        # set index
        cell.setIndex(i)
    return(cells)

def prepareCells(cells, width):
    cells = initializeCells(cells, width)
    return(cells)

class MapParser:
    gridMap = Map()

    def parseLine(self, line):
        cells = []
        for c in line:
            parseAction = symbolToEnum(c)
            if parseAction != ParseAction.NEW_LINE:
                entity = symbolToEntity(c)
                cells.append(entity)
        return cells

    def parseMap(self, fname):
        height = 0
        width = None
        cells = []
        with open(fname) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                if width != None and (len(line) - 1)!= width:
                    raise Exception("Input width inconsistent")
                width = len(line) - 1
                height += 1
                rowCells = self.parseLine(line)
                cells.extend(rowCells)
        self.gridMap.setWidth(width)
        self.gridMap.setHeight(height)
        cells = prepareCells(cells, width)
        self.gridMap.setCells(cells)
        print("Parsed map of dim: " + str(height) + "x" + str(width))
        return(self.gridMap)
