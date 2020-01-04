from Perceptions import Perceptions
from DataItem import DataItem

class Perception(DataItem):

    def __init__(self, perceptionType):
        self.perceptionType = perceptionType

    def isPerception(self):
        return True

    def isAction(self):
        return False

    def getType(self):
        return self.perceptionType
