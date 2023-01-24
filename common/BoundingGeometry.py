import math
from typing import Optional

import miniball
import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from common.Box import Box
from common.Sphere import Sphere


class BoundingGeometry:
    _points: list[list[float]]
    _sphere: Optional[Sphere]
    _box: Optional[Box]

    def __init__(self, points=None):
        self._points = []
        if points is not None:
            self.extendByPoints(points)

    def getSphere(self) -> Sphere:
        if self._sphere is None:
            self._computeBoundingSphere()
        return self._sphere

    def getBox(self) -> Box:
        if self._box is None:
            self._computeBoundingBox()
        return self._box

    def _computeBoundingSphere(self) -> None:
        if len(self._points) == 0:
            raise Exception("missing points")

        try:
            hull = ConvexHull(self._points)
            vertices = hull.vertices
        except QhullError:
            vertices = range(len(self._points))

        points = np.array([self._points[i] for i in vertices])

        try:
            rng = np.random.default_rng(seed=0)
            center, r2 = miniball.get_bounding_ball(points, rng=rng)
            self._sphere = Sphere(center.tolist(), math.sqrt(r2))
        except LinAlgError:
            self._computeBoundingSphereFallback(points)

    def _computeBoundingSphereFallback(self, convexHullPoints: ndarray) -> None:
        center = np.mean([np.min(convexHullPoints, axis=0), np.max(convexHullPoints, axis=0)], axis=0)
        radius = max(np.linalg.norm(np.subtract(convexHullPoints, center), axis=1))
        self._sphere = Sphere(center.tolist(), radius)

    def _computeBoundingBox(self) -> None:
        if len(self._points) == 0:
            raise Exception("missing points")

        points = np.array(self._points)
        self._box = Box(np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist())

    def _resetSphereAndBox(self) -> None:
        self._sphere = None
        self._box = None

    def extendByPoint(self, point: list[float]) -> None:
        self._points.append(point)
        self._resetSphereAndBox()

    def extendByPoints(self, points: list[list[float]]) -> None:
        self._points.extend(points)
        self._resetSphereAndBox()

    def extendBySphere(self, center: list[float], radius: float) -> None:
        # TODO for sphere this is not correct. However it is ensured that the calculated bounding geometry is not smaller than the actual bounding geometry
        minVertex = np.subtract(center, [radius]).tolist()
        maxVertex = np.add(center, [radius]).tolist()
        self.extendByPoints([minVertex, maxVertex])

    def extendByBoundingGeometry(self, boundingGeometry: "BoundingGeometry") -> None:
        self.extendByPoints(boundingGeometry._points)
