from typing import Optional

from common.Util import Util


class EntityItem:
    archetypeName: str
    position: list[float]
    scale: list[float]
    rotation: list[float]
    lodDistance: float
    childLodDist: Optional[float]
    parentIndex: Optional[int]
    numChildren: Optional[int]
    lodLevel: Optional[str]
    flags: Optional[int]

    def __init__(self, archetypeName: str, position: list[float], scale: list[float], rotation: list[float], lodDistance: float, childLodDist: Optional[float] = None,
            parentIndex: Optional[int] = None, numChildren: Optional[int] = None, lodLevel: Optional[str] = None, flags: Optional[int] = None):
        self.archetypeName = archetypeName
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.lodDistance = lodDistance
        self.childLodDist = childLodDist
        self.parentIndex = parentIndex
        self.numChildren = numChildren
        self.lodLevel = lodLevel
        self.flags = flags

    def applyTransformationTo(self, vertex: list[float]) -> list[float]:
        return Util.applyTransformation(vertex, self.rotation, self.scale, self.position)
