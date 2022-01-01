import math
import os
import random
import shutil
from typing import Any

import numpy as np
from natsort import natsorted
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

from common import Box, Sphere


class Util:
    MIN_LOD_DISTANCE = 10

    @staticmethod
    def floatToStr(val: float) -> str:
        return "{:.8f}".format(val)

    @staticmethod
    def calculateFurthestDistance(coords: list[list[float]]) -> float:
        if len(coords) == 0:
            return -1
        elif len(coords) == 1:
            return 0
        elif len(coords) == 2:
            return math.dist(coords[0], coords[1])
        elif len(coords) == 3:
            return max(math.dist(coords[0], coords[1]), math.dist(coords[0], coords[2]), math.dist(coords[1], coords[2]))

        points = np.array(coords)
        hull = ConvexHull(points)
        maxDistance = -1
        for index1 in hull.vertices:
            point1 = points[index1]
            for index2 in hull.vertices:
                point2 = points[index2]
                maxDistance = max(maxDistance, np.linalg.norm(point1 - point2))

        return maxDistance

    @staticmethod
    def performClustering(points: list[list[float]], maxPoints: int, maxFurthestDistance: float) -> (Any, list[float]):
        X = np.array(points)

        maxClusterSize = -1
        furthestDistances = [math.inf]
        clusters = None
        numClusters = 0 if maxPoints <= 0 else math.ceil(len(points) / maxPoints) - 1
        while maxClusterSize < 0 or 0 < maxPoints < maxClusterSize or max(furthestDistances) > maxFurthestDistance:
            numClusters += 1
            model = KMeans(n_clusters=numClusters, n_init=200, max_iter=5000, random_state=0, algorithm="full")
            clusters = model.fit_predict(X)

            maxClusterSize = -1
            furthestDistances = [0] * numClusters
            for cluster in np.unique(clusters):
                clusterEntries = np.where(clusters == cluster)

                maxClusterSize = max(maxClusterSize, len(clusterEntries[0]))

                furthestDistances[cluster] = Util.calculateFurthestDistance(X[clusterEntries[0]])

        return clusters, furthestDistances

    @staticmethod
    def calculateLodDistance(boundingBox: Box, boundingSphere: Sphere, scale: list[float], hasParent: bool) -> float:
        scaledBoundingBox = boundingBox.getScaled(scale)
        sizes = scaledBoundingBox.getSizes()
        # diagonals = scaledBoundingBox.getPlaneDiagonals()
        radius = boundingSphere.radius

        #areasDiagPlanes = []
        areasPlanes = []
        areasEllipses = []
        for i in range(3):
            # areasDiagPlanes.append(sizes[i] * diagonals[i])
            areasPlanes.append(sizes[i] * sizes[(i + 1) % 3])
            areasEllipses.append(math.pi * radius * scale[i] * radius * scale[(i + 1) % 3])  # area of ellipse is pi * radius1 * radius2

        noParentMultiplier = 1 if hasParent else 1.75

        # lodDistBBoxBSphere = math.log(1 + math.sqrt(min(max(areasDiagPlanes), max(areasEllipses)))) * 90
        # lodDistBBoxBSphere = math.log(1 + min(max(areasDiagPlanes), max(areasEllipses))) * 48
        lodDistBBoxBSphere = math.log(1 + noParentMultiplier * min(max(areasPlanes), max(areasEllipses))) * 48

        return max(lodDistBBoxBSphere, Util.MIN_LOD_DISTANCE)
        # scaledLodDist = math.sqrt(max(scale)) * self.lodDist
        # return max(scaledLodDist, lodDistBBoxBSphere)

    @staticmethod
    def angleToStr(angleInRad):
        return str(round(math.degrees(angleInRad), 2))

    @staticmethod
    def calculateMaxTilt(height: float) -> float:
        if height > 100:
            result = 2.5
        elif height > 75:
            result = 3
        elif height > 50:
            result = 4
        elif height > 25:
            result = 5
        elif height > 10:
            result = 6
        elif height > 4:
            result = 8
        else:
            result = 10

        return random.uniform(math.radians(result * 0.85), math.radians(result * 1.15))

    @staticmethod
    def getListOfFiles(inputDir: str, filter):
        result = []
        for filename in natsorted(os.listdir(inputDir)):
            if os.path.isfile(os.path.join(inputDir, filename)) and filter(filename):
                result.append(filename)

        return result

    @staticmethod
    def copyFiles(inputDir: str, outputDir: str, filter):
        for filename in Util.getListOfFiles(inputDir, filter):
            shutil.copyfile(os.path.join(inputDir, filename), os.path.join(outputDir, filename))

    @staticmethod
    def readFile(path: str) -> str:
        file = open(path, 'r')
        content = file.read()
        file.close()
        return content
