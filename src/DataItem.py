from abc import ABC, abstractmethod

class DataItem(ABC):

    @abstractmethod
    def isPerception(self):
        pass

    @abstractmethod
    def isAction(self):
        pass
