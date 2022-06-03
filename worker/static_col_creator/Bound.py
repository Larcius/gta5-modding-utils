class Bound:
    phBounds: list[list[str]]
    childFlagItems: list[list[str]]

    def __init__(self, phBounds: list[list[str]], childFlagItems: list[list[str]]):
        if len(phBounds) != len(childFlagItems):
            raise Exception("number of phBounds and childFlagItems must be equal")

        self.phBounds = phBounds
        self.childFlagItems = childFlagItems

    def getNumChildren(self) -> int:
        return len(self.phBounds)
