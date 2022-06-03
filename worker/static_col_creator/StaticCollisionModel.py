from worker.static_col_creator.Bound import Bound


class StaticCollisionModel:
    default: Bound
    ma: Bound
    high: Bound

    def __init__(self, linesPhBounds: list[str], linesChildFlagItems: list[str]):
        phBounds = StaticCollisionModel.parseLines(linesPhBounds)
        childFlagItems = StaticCollisionModel.parseLines(linesChildFlagItems)

        if len(phBounds) != len(childFlagItems):
            raise Exception("number of phBounds and childFlagItems must be equal")

        phBoundsDefault = []
        phBoundsMa = []
        phBoundsHigh = []
        childFlagItemsDefault = []
        childFlagItemsMa = []
        childFlagItemsHigh = []

        for i in range(len(phBounds)):
            phBound = phBounds[i]
            childFlagItem = childFlagItems[i]
            if childFlagItem[2] == "			Flags1 MAP_WEAPON\n":
                phBoundsHigh.append(phBound)
                childFlagItemsHigh.append(childFlagItem)
            elif childFlagItem[2] == "			Flags1 FORMATS_EMPTY_FLAGS\n":
                phBoundsMa.append(phBound)
                childFlagItemsMa.append(childFlagItem)
            else:
                phBoundsDefault.append(phBound)
                childFlagItemsDefault.append(childFlagItem)

        self.default = Bound(phBoundsDefault, childFlagItemsDefault)
        self.ma = Bound(phBoundsMa, childFlagItemsMa)
        self.high = Bound(phBoundsHigh, childFlagItemsHigh)

    @staticmethod
    def parseLines(lines: list[str]) -> list[list[str]]:
        if not lines:
            return []

        result = []
        cur = []
        for line in lines:
            cur.append(line)
            if line == "		}\n":
                result.append(cur)
                cur = []

        if cur:
            raise Exception("got content after last \t\t}")

        return result
