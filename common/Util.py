import math
import os
import random
import shutil
import struct
from typing import Any, Callable, Optional

import numpy as np
import transforms3d
from datetime import datetime
from natsort import natsorted
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial.qhull import QhullError
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from common import Box, Sphere


class Util:
    MIN_LOD_DISTANCE = 10

    @staticmethod
    def floatToStr(val: float) -> str:
        return "{:.8f}".format(val)

    @staticmethod
    def vectorToStr(vertex: list[float]) -> str:
        converted = []
        for i in range(len(vertex)):
            converted.append(Util.floatToStr(vertex[i]))
        return " ".join(converted)

    @staticmethod
    def calculateFurthestDistance(coords: list[list[float]]) -> float:
        coords = np.unique(coords, axis=0)

        if len(coords) == 0:
            return -1
        elif len(coords) == 1:
            return 0
        elif len(coords[0]) == 1:
            return max(coords) - min(coords)
        elif len(coords) < 11:
            # this is only mandatory for len(coords) < 5 because ConvexHull needs at least 5 points
            # however if there are only a few points then just compute the pairwise distances
            return max(pdist(coords))

        # for a large set of points do not compute the pairwise distances but first calculate the convex hull and
        # just compute the maximum of the pairwise distances from that hull vertices.
        # also don't use pdist here since we are only interested in the maximum distances and hence there is no need to
        # actually create a list of size n*(n-1)/2 (where n is the number of vertices of the convex hull)
        points = np.array(coords)
        try:
            hull = ConvexHull(points)
            vertices = hull.vertices
        except QhullError:
            vertices = range(len(coords))

        maxDistance = 0
        for index1 in vertices:
            point1 = coords[index1]
            for index2 in vertices:
                if index1 >= index2:
                    continue

                point2 = coords[index2]
                maxDistance = max(maxDistance, math.dist(point1, point2))

        return maxDistance

    @staticmethod
    def _performClustering(X: np.ndarray, numClusters: int, unevenClusters: bool) -> (Any, float, list[float]):
        print("\t\tcalculating clustering of " + str(numClusters) + " clusters")
        if unevenClusters:
            model = AgglomerativeClustering(n_clusters=numClusters, linkage="average")
        else:
            model = MiniBatchKMeans(n_clusters=numClusters, random_state=0, reassignment_ratio=0, n_init=10)
        clusters = model.fit_predict(X)

        clusters = Util._fixClusterLabels(clusters, X)

        maxClusterSize = -1
        furthestDistances = [0] * numClusters
        for cluster in np.unique(clusters):
            clusterEntries = np.where(clusters == cluster)

            maxClusterSize = max(maxClusterSize, len(clusterEntries[0]))

            furthestDistances[cluster] = Util.calculateFurthestDistance(X[clusterEntries[0]])

        return clusters, maxClusterSize, furthestDistances

    @staticmethod
    def _fixClusterLabels(clusters, X: np.ndarray):
        uniqueClusters = np.unique(clusters)

        dim = X.shape[1]

        minVertex = X.min(axis=0)
        maxVertex = X.max(axis=0)
        extents = maxVertex - minVertex

        sideLengths = np.zeros(dim)
        centers = []
        for cluster in uniqueClusters:
            clusterEntries = np.where(clusters == cluster)
            points = X[clusterEntries[0]]
            centers.append(sum(points) / len(clusterEntries[0]))
            clusterExtents = points.max(axis=0) - points.min(axis=0)
            sideLengths += clusterExtents
        centers = np.array(centers)

        lexsortCriteria = []
        if dim == 1:
            lexsortCriteria.append(centers[:, 0])
        else:
            sideLengths = np.maximum(np.ones(dim), sideLengths)  # prevent dividing by zero if there is a degenerated dimension
            numSteps = np.ceil(extents / sideLengths * len(uniqueClusters))
            extents = np.maximum(np.ones(dim), extents)  # prevent dividing by zero if there is a degenerated dimension

            discreteCenters = []
            for center in centers:
                discreteCenter = np.floor(numSteps * (center - minVertex) / extents)
                discreteCenters.append(discreteCenter)
            discreteCenters = np.array(discreteCenters)

            for i in range(dim):
                j = dim - i - 1
                if j == 0:
                    j = 1
                elif j == 1:
                    j = 0
                lexsortCriteria.append(centers[:, j] * (-1 if j == 1 else 1))
            for i in range(dim):
                j = dim - i - 1
                if j == 0:
                    j = 1
                elif j == 1:
                    j = 0
                lexsortCriteria.append(discreteCenters[:, j] * (-1 if j == 1 else 1))

        order = np.lexsort(lexsortCriteria)
        orderMapping = {}
        for i in range(len(order)):
            orderMapping[order[i]] = i
        mapping = {}
        for i in range(len(uniqueClusters)):
            mapping[uniqueClusters[i]] = orderMapping[i]

        newClusters = []
        for i in range(len(clusters)):
            origCluster = clusters[i]
            newClusters.append(mapping[origCluster])

        return newClusters

    @staticmethod
    def performClusteringFixedPolygon(points: list[list[float]], polygon: list[list[float]]) -> Any:
        polygon = Polygon(polygon)
        clusters = []
        hasPointInside = False
        hasPointOutside = False
        for p in points:
            point = Point(p[0], p[1])
            if polygon.contains(point):
                hasPointInside = True
                c = 0
            else:
                hasPointOutside = True
                c = 1
            clusters.append(c)

        if hasPointInside and hasPointOutside:
            return clusters
        else:
            return np.zeros(len(points))

    @staticmethod
    def performClusteringFixedNumClusters(points: list[list[float]], numClusters: int, unevenClusters: bool = False) -> (Any, float, list[float]):
        X = np.array(points)
        return Util._performClustering(X, numClusters, unevenClusters)

    @staticmethod
    def performClustering(points: list[list[float]], maxPoints: int, maxFurthestDistance: float, unevenClusters: bool = False) -> (Any, list[float]):
        numPoints = len(points)
        if numPoints == 1:
            return np.array([0]), [0]

        X = np.array(points)

        largestNonValidNumClusters = 0
        smallestValidNumClusters = None
        clustersForSmallestValidNumClusters = None
        furthestDistancesForSmallestValidNumClusters = None

        numClusters = 1 if maxPoints <= 0 else math.ceil(len(points) / maxPoints)
        while largestNonValidNumClusters + 1 != smallestValidNumClusters:
            clusters, maxClusterSize, furthestDistances = Util._performClustering(X, numClusters, unevenClusters)

            furthestDistanceWeightedMean = 0
            for cluster in np.unique(clusters):
                clusterEntries = np.where(clusters == cluster)
                clusterSize = len(clusterEntries[0])
                furthestDistanceWeightedMean += clusterSize * furthestDistances[cluster]
            furthestDistanceWeightedMean /= max(1, numPoints)

            exceededLimits = 0 < maxPoints < maxClusterSize or furthestDistanceWeightedMean > maxFurthestDistance
            if exceededLimits:
                largestNonValidNumClusters = numClusters
                if smallestValidNumClusters is None:
                    nextNumClusters = numClusters + 1
                    if 0 < maxPoints < maxClusterSize:
                        nextNumClusters = max(nextNumClusters, numClusters + math.ceil(maxClusterSize / maxPoints))
                    if max(furthestDistances) > maxFurthestDistance:
                        ratio = max(furthestDistances) / maxFurthestDistance
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
    def getListOfFiles(inputDir: str, filter: Optional[Callable[[str], bool]]):
        result = []
        for filename in natsorted(os.listdir(inputDir)):
            if os.path.isfile(os.path.join(inputDir, filename)) and (filter is None or filter(filename)):
                result.append(filename)

        return result

    @staticmethod
    def copyFiles(inputDir: str, outputDir: str, filter: Optional[Callable[[str], bool]]):
        for filename in Util.getListOfFiles(inputDir, filter):
            shutil.copyfile(os.path.join(inputDir, filename), os.path.join(outputDir, filename))

    @staticmethod
    def readFile(path: str) -> str:
        file = open(path, 'r')
        content = file.read()
        file.close()
        return content

    @staticmethod
    def writeFile(path: str, content: str):
        file = open(path, 'w')
        file.write(content)
        file.close()

    @staticmethod
    def calculateAngle(vertexMiddle: list[float], vertex1: list[float], vertex2: list[float]) -> float:
        unitVector1 = Util.normalize(np.subtract(vertex1, vertexMiddle))
        unitVector2 = Util.normalize(np.subtract(vertex2, vertexMiddle))

        return np.arccos(np.dot(unitVector1, unitVector2))

    @staticmethod
    def normalize(vector: list[float]) -> list[float]:
        norm = np.linalg.norm(vector)
        if abs(norm) < 1e-8:
            return vector
        else:
            return vector / norm

    @staticmethod
    def determinePrefixBundles(names: list[str]) -> list[str]:
        bundlePrefixes = []
        numMatches = 0
        candidate = None
        for name in natsorted(names):
            if candidate is None:
                candidate = name

            parts = [candidate.rstrip("_"), "", ""]
            while not (parts[0] == "" or name == parts[0] or name.startswith(parts[0] + "_")):
                if "_" not in parts[0]:
                    parts = [""]
                else:
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

        bundlePrefixes = natsorted(bundlePrefixes)

        # remove prefixes of prefixes
        removedPrefix = False
        i = 0
        while i + 1 < len(bundlePrefixes):
            if bundlePrefixes[i + 1].startswith(bundlePrefixes[i]):
                removedPrefix = True
                bundlePrefixes.pop(i)
            else:
                i += 1

        if not removedPrefix:
            return bundlePrefixes

        # add names that are no longer covered by any prefix
        for name in names:
            foundSuitablePrefix = False
            for prefix in bundlePrefixes:
                if name.startswith(prefix):
                    foundSuitablePrefix = True
                    break

            if not foundSuitablePrefix:
                bundlePrefixes.append(name)

        return natsorted(bundlePrefixes)

    @staticmethod
    def getNowInIsoFormat() -> str:
        return datetime.now().astimezone().replace(microsecond=0).isoformat()

    @staticmethod
    def applyRotation(vertex: list[float], rotation: list[float]) -> list[float]:
        return transforms3d.quaternions.rotate_vector(vertex, rotation)

    @staticmethod
    def applyTransformation(vertex: list[float], rotation: list[float], scaling: list[float], translation: list[float]) -> list[float]:
        return np.add(np.multiply(Util.applyRotation(vertex, rotation), scaling), translation).tolist()

    @staticmethod
    def hashFloat(val: float) -> int:
        return hash(struct.pack("f", val))
