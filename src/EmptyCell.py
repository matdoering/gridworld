from Cell import Cell

class EmptyCell(Cell):
    neighborCells = []

    def canBeEntered(self):
        return True

    def __str__(self):
        if self.hasActor():
            return super().__str__()
        else:
            return " "
