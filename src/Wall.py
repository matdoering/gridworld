from Cell import Cell

class Wall(Cell):

    def canBeEntered(self):
        return False

    def isWall(self):
        return True

    def __str__(self):
        if self.hasActor():
            return super().__str__()
        else:
            return "#"
