import math
import os
import random
import shutil
from typing import Any

import numpy as np
from natsort import natsorted
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from common import Box, Sphere


class Util:
    MIN_LOD_DISTANCE = 10

    @staticmethod
    def floatToStr(val: float) -> str:
        return "{:.8f}".format(val)

    @staticmethod
    def calculateFurthestDistance(coords: list[list[float]]) -> float:
        coords = np.unique(coords, axis=0)

        if len(coords) == 0:
            return -1
        elif len(coords) == 1:
            return 0
        elif len(coords) < 11:
            # this is only mandatory for len(coords) < 5 because ConvexHull needs at least 5 points
            # however if there are only a few points then just compute the pairwise distances
            return max(pdist(coords))

        # for a large set of points do not compute the pairwise distances but first calculate the convex hull and
        # just compute the maximum of the pairwise distances from that hull vertices.
        # also don't use pdist here since we are only interested in the maximum distances and hence there is no need to
        # actually create a list of size n*(n-1)/2 (where n is the number of vertices of the convex hull)
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
    def _performClustering(X: np.ndarray, numClusters: int) -> (Any, float, list[float]):
        print("\t\tcalculating clustering of " + str(numClusters) + " clusters")
        model = KMeans(n_clusters=numClusters, n_init=200, max_iter=5000, random_state=0, algorithm="full")
        clusters = model.fit_predict(X)

        maxClusterSize = -1
        furthestDistances = [0] * numClusters
        for cluster in np.unique(clusters):
            clusterEntries = np.where(clusters == cluster)

            maxClusterSize = max(maxClusterSize, len(clusterEntries[0]))

            furthestDistances[cluster] = Util.calculateFurthestDistance(X[clusterEntries[0]])

        return clusters, maxClusterSize, furthestDistances

    @staticmethod
    def performClusteringFixedNumClusters(points: list[list[float]], numClusters: int) -> (Any, float, list[float]):
        X = np.array(points)
        return Util._performClustering(X, numClusters)

    @staticmethod
    def performClustering(points: list[list[float]], maxPoints: int, maxFurthestDistance: float) -> (Any, list[float]):
        X = np.array(points)

        largestNonValidNumClusters = 0
        smallestValidNumClusters = None
        clustersForSmallestValidNumClusters = None
        furthestDistancesForSmallestValidNumClusters = None

        numClusters = 1 if maxPoints <= 0 else math.ceil(len(points) / maxPoints)
        while largestNonValidNumClusters + 1 != smallestValidNumClusters:
            clusters, maxClusterSize, furthestDistances = Util._performClustering(X, numClusters)

            exceededLimits = 0 < maxPoints < maxClusterSize or max(furthestDistances) > maxFurthestDistance
            if exceededLimits:
                largestNonValidNumClusters = numClusters
                if smallestValidNumClusters is None:
                    nextNumClusters = numClusters + 1
                    if 0 < maxPoints < maxClusterSize:
                        nextNumClusters = max(nextNumClusters, numClusters + math.ceil(maxClusterSize / maxPoints))
                    if max(furthestDistances) > maxFurthestDistance:
                        ratio = max(furthestDistances) / maxFurthestDistance
                        if numClusters == 1:
                            # assuming that the points are more likely being distributed on a plane than on a line raise this to the power of 2
                            ratio **= 2
                        nextNumClusters = max(nextNumClusters, math.ceil(numClusters * ratio))
                    # ensure that there are at most as many clusters as points
                    numClusters = min(len(points), nextNumClusters)
                else:
                    numClusters = math.ceil((largestNonValidNumClusters + smallestValidNumClusters) / 2)
            else:
                clustersForSmallestValidNumClusters = clusters
                furthestDistancesForSmallestValidNumClusters = furthestDistances
                smallestValidNumClusters = numClusters
                numClusters = math.floor((largestNonValidNumClusters + smallestValidNumClusters) / 2)

        print("\t\tfound valid clustering consisting of " + str(len(np.unique(clustersForSmallestValidNumClusters))) + " clusters")

        return clustersForSmallestValidNumClusters, furthestDistancesForSmallestValidNumClusters

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

    @staticmethod
    def calculateAngle(vertexMiddle: list[float], vertex1: list[float], vertex2: list[float]) -> float:
        vector1 = np.subtract(vertex1, vertexMiddle)
        vector2 = np.subtract(vertex2, vertexMiddle)

        unitVector1 = vector1 / np.linalg.norm(vector1)
        unitVector2 = vector2 / np.linalg.norm(vector2)
        return np.arccos(np.dot(unitVector1, unitVector2))

    @staticmethod
    def determinePrefixBundles(names: list[str]) -> list[str]:
        bundlePrefixes = []
        numMatches = 0
        candidate = None
        for name in natsorted(names):
            if candidate is None:
                candidate = name

            parts = [candidate.rstrip("_"), "", ""]
            while not (name == parts[0] or name.startswith(parts[0] + "_")):
                parts = parts[0].rpartition("_")

            newCandidate = parts[0]

            if newCandidate == "":
                if numMatches > 0:
                    bundlePrefixes.append(candidate)
                numMatches = 1
                candidate = name
            elif newCandidate == candidate:
                numMatches += 1
            else:
                if numMatches > 1:
                    bundlePrefixes.append(candidate)
                    numMatches = 1
                    candidate = name
                else:
                    numMatches += 1
                    candidate = newCandidate

        if candidate != "" and numMatches > 0:
            bundlePrefixes.append(candidate)

        return bundlePrefixes
