import numpy as np


class Ellipsoid:
    center: list[float]
    radii: list[float]

    def __init__(self, center: list[float], radii: list[float]):
        self.center = center
        self.radii = radii

    def getTranslated(self, translation: list[float]) -> "Ellipsoid":
        translatedCenter = np.add(self.center, translation)
        return Ellipsoid(translatedCenter, self.radii)

    def getScaled(self, scale: list[float]) -> "Ellipsoid":
        scaledRadii = np.multiply(self.radii, scale)
        return Ellipsoid(self.center.copy(), scaledRadii.tolist())

#    def getEnclosingBox(self) -> Box:
#        return Box(self.center.copy(), self.center.copy()).getExtended(self.radii)

#    def getEnclosingSphere(self) -> Sphere:
#        return Sphere(self.center.copy(), max(self.radii))
