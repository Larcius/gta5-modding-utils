import math
import os
import shutil
import re
from string import digits
from typing import Any, Callable, Optional

import numpy as np
import transforms3d
from datetime import datetime
from natsort import natsorted
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial.qhull import QhullError
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
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
    def _performClustering(X: np.ndarray, numClusters: Optional[int], distanceThreshold: Optional[float], unevenClusters: bool) -> (Any, int, list[float]):
        numPoints = X.shape[0]
        if numClusters is None:
            print("\t\tcalculating clustering using distance threshold " + str(distanceThreshold) + " for " + str(numPoints) + " points")
        else:
            print("\t\tcalculating clustering of " + str(numClusters) + " clusters for " + str(numPoints) + " points")

        if numClusters == 1:
            clusters = np.zeros(numPoints, dtype=int)
        elif numClusters == numPoints:
            clusters = np.arange(numPoints, dtype=int)
        else:
            if unevenClusters:
                model = AgglomerativeClustering(n_clusters=numClusters, distance_threshold=distanceThreshold, linkage="complete")
            else:
                model = KMeans(n_clusters=numClusters, random_state=0, n_init=10)
            clusters = model.fit_predict(X)

            clusters = Util._fixClusterLabels(clusters, X)

        numClusters = max(clusters) + 1

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
    def performClusteringFixedNumClusters(points: list[list[float]], numClusters: int, unevenClusters: bool = False) -> (Any, int, list[float]):
        X = np.array(points)
        return Util._performClustering(X, numClusters, None, unevenClusters)

    @staticmethod
    def performClusteringMaxFurthestDistance(points: list[list[float]], maxFurthestDistance: float) -> (Any, int, list[float]):
        X = np.array(points)
        return Util._performClustering(X, None, maxFurthestDistance, True)

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

        if maxPoints > 0:
            numClusters = math.ceil(numPoints / maxPoints)
        else:
            numClusters = 1

        while largestNonValidNumClusters + 1 != smallestValidNumClusters:
            clusters, maxClusterSize, furthestDistances = Util._performClustering(X, numClusters, None, unevenClusters)

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
                    numClusters = min(numPoints, nextNumClusters)
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
        return str(round(math.degrees(angleInRad), 2)) + "°"

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

        return math.radians(result)

    @staticmethod
    def getListOfFiles(inputDir: str, filter: Optional[Callable[[str], bool]] = None):
        result = []
        for filename in natsorted(os.listdir(inputDir)):
            if os.path.isfile(os.path.join(inputDir, filename)) and (filter is None or filter(filename)):
                result.append(filename)

        return result

    @staticmethod
    def copyFiles(inputDir: str, outputDir: str, filter: Optional[Callable[[str], bool]] = None):
        for filename in Util.getListOfFiles(inputDir, filter):
            Util.copyFile(inputDir, outputDir, filename)

    @staticmethod
    def copyFile(inputDir: str, outputDir: str, filename: str, filenameDestination: Optional[str] = None):
        destination = os.path.join(outputDir, filename if filenameDestination is None else filenameDestination)
        if not os.path.isfile(destination):
            shutil.copyfile(os.path.join(inputDir, filename), destination)

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
    def getFilenameFromMapname(mapName: str) -> str:
        return mapName + ".ymap.xml"

    @staticmethod
    def getMapnameFromFilename(filename: str) -> Optional[str]:
        if filename.endswith(".ymap.xml"):
            return filename[:-9]
        else:
            return None

    @staticmethod
    def findAvailableMapName(dir: str, mapName: str, suffix: str, ensureSuffix: bool) -> str:
        if ensureSuffix:
            newMapName = re.sub(suffix + "\\d*$", "", mapName) + suffix
        else:
            newMapName = mapName

        if os.path.exists(os.path.join(dir, Util.getFilenameFromMapname(newMapName))):
            newMapName = re.sub(suffix + "\\d*$", "", newMapName) + suffix
            i = -1
            while os.path.exists(os.path.join(dir, Util.getFilenameFromMapname(newMapName + ("" if i < 0 else str(i))))):
                i += 1
            if i >= 0:
                newMapName += str(i)
        return newMapName

    @staticmethod
    def determinePrefixBundles(names: list[str]) -> list[str]:
        prefixes = set()
        for name in names:
            prefixes.add(name.rstrip(digits))

        return natsorted(list(prefixes))

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
        return hash(round(val, ndigits=5))
