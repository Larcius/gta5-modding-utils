import numpy as np

from common.Ellipsoid import Ellipsoid


# TODO extends Ellipsoid
class Sphere:
    @staticmethod
    def createUnitSphere():
        return Sphere([0] * 3, 0.5)

    center: list[float]
    radius: float

    def __init__(self, center: list[float], radius: float):
        self.center = center
        self.radius = radius

    def getTranslated(self, translation: list[float]) -> "Sphere":
        translatedCenter = np.add(self.center, translation)
        return Sphere(translatedCenter.tolist(), self.radius)

    def getScaled(self, scale: list[float]) -> Ellipsoid:
        scaledRadii = np.multiply([self.radius] * 3, scale)
        return Ellipsoid(self.center.copy(), scaledRadii.tolist())

    def getEllipsoid(self) -> Ellipsoid:
        return Ellipsoid(self.center.copy(), [self.radius] * 3)

#    def getEnclosingBox(self) -> Box:
#        return Box(self.center.copy(), self.center.copy()).getExtended([self.radius] * 3)
