import numpy as np
import math

from common.Sphere import Sphere


class Box:
    @staticmethod
    def createReversedInfinityBox() -> "Box":
        return Box([math.inf] * 3, [-math.inf] * 3)

    @staticmethod
    def createUnitBox() -> "Box":
        return Box([-0.5] * 3, [0.5] * 3)

    min: list[float]
    max: list[float]

    def __init__(self, minVertex: list[float], maxVertex: list[float]):
        self.min = minVertex
        self.max = maxVertex

    def getCenter(self) -> list[float]:
        return np.divide(np.add(self.min, self.max), [2]).tolist()

    # returns the diagonal (distance between points min and max
    def getDiagonal(self) -> float:
        return math.dist(self.min, self.max)

    # returns diagonals of YZ-, XZ-, XY-planes (in that order)
    def getPlaneDiagonals(self) -> list[float]:
        diagonals = []
        for i in range(3):
            diagonals.append(math.dist([self.min[(i + 1) % 3], self.min[(i + 1) % 3]], [self.max[(i + 2) % 3], self.max[(i + 1) % 3]]))
        return diagonals

    def getSizes(self) -> list[float]:
        return np.subtract(self.max, self.min).tolist()

    def getTranslated(self, translation: list[float]) -> "Box":
        translatedMin = np.add(self.min, translation)
        translatedMax = np.add(self.max, translation)
        return Box(translatedMin.tolist(), translatedMax.tolist())

    def getScaled(self, scale: list[float]) -> "Box":
        scaledMin = np.multiply(self.min, scale)
        scaledMax = np.multiply(self.max, scale)
        return Box(scaledMin.tolist(), scaledMax.tolist())

    def getExtended(self, extents: list[float]) -> "Box":
        extendedMin = np.subtract(self.min, extents)
        extendedMax = np.add(self.max, extents)
        return Box(extendedMin.tolist(), extendedMax.tolist())

    def extendByPoint(self, point: list[float]):
        self.min = np.minimum(self.min, point).tolist()
        self.max = np.maximum(self.max, point).tolist()

    def getEnclosingSphere(self) -> Sphere:
        return Sphere(self.getCenter().copy(), self.getDiagonal() / 2)

    def isValid(self):
        for i in range(3):
            if not math.isfinite(self.min[i]) or not math.isfinite(self.max[i]) or self.min[i] > self.max[i]:
                return False

        return True
