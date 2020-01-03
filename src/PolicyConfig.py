
def getDefaultPolicyConfig():
    return PolicyConfig(setStickyWalls = False)

class StickyWallConfig:

    def __init__(self, p):
        self.p = p # probability of sticking to wall when wall is adjoining agent

class PolicyConfig:
    def __init__(self, setStickyWalls):
        if setStickyWalls:
            self.stickyWallConfig = StickyWallConfig(0.40)
        else:
            self.stickyWallConfig = StickyWallConfig(0.0)

    def getStickyWallConfig(self):
        return self.stickyWallConfig



