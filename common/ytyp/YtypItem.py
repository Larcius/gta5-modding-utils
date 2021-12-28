from common.Box import Box
from common.Sphere import Sphere
from common.Util import Util


class YtypItem:
    lodDist: float
    boundingBox: Box
    boundingSphere: Sphere
    parent: str

    def __init__(self, lodDist: float, boundingBox: Box, boundingSphere: Sphere, parent: str):
        self.lodDist = lodDist
        self.boundingBox = boundingBox
        self.boundingSphere = boundingSphere
        self.parent = parent

    def getLodDistance(self, scale: list[float], hasParent: bool) -> float:
        return Util.calculateLodDistance(self.boundingBox, self.boundingSphere, scale, hasParent)
