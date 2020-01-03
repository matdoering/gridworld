from EmptyCell import EmptyCell

class GoalCell(EmptyCell):

    def __init__(self):
        super().__init__()
        self.containsGoal = True

    def __str__(self):
        if self.hasActor():
            return super().__str__()
        else:
            return "X"
