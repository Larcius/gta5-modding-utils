import math
from re import Match
from natsort import natsorted
import numpy as np
import os
import re

from common.Util import Util
from common.ymap.Ymap import Ymap
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class Reducer:
    inputDir: str
    outputDir: str

    defaultYmapPart: str
    ytypItems: dict[str, YtypItem]
    prefix: str
    reduceFactor: float
    adaptScaling: bool

    _PATTERN = re.compile(
        '([\t ]*<Item type="CEntityDef">' +
        '\\s*<archetypeName>([^<]+)</archetypeName>' +
        '(?:\\s*<[^/].*>)*?' +
        '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
        '(?:\\s*<[^/].*>)*?' +
        '\\s*<scaleXY value=")([^"]+)("/>' +
        '\\s*<scaleZ value=")([^"]+)("/>' +
        '(?:\\s*<[^/].*>)*?' +
        '\\s*</Item>[\r\n]+)'
    )

    def __init__(self, inputDir: str, outputDir: str, prefix: str, reduceFactor: float, adaptScaling: bool):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix
        self.reduceFactor = reduceFactor
        self.adaptScaling = adaptScaling

    def run(self):
        print("running reducer...")
        self.readYtyps()
        self.createOutputDir()
        self.processFiles()
        self.copyOthers()
        print("reducer DONE")

    def readYtyps(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "../..", "resources", "ytyp"))

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)

    def calculatePointsToKeep(self, points: list[list[float]]) -> list[int]:
        numPoints = len(points)
        if numPoints == 0:
            return []

        numPointsToKeep = math.ceil(numPoints * self.reduceFactor)

        if numPointsToKeep == 0:
            return [0] * numPoints

        clustering, unused, unused = Util.performClusteringFixedNumClusters(points, numPointsToKeep, True)

        clusterSizes = np.bincount(clustering)

        clustersMidpoint = self.calculateClustersMidpoint(points, clustering)

        closestPointToClusterMidpoint = self.calculateClosestPointToClusterMidpoint(points, clustering, clustersMidpoint)

        pointsToKeep = []
        for i in range(numPoints):
            cluster = clustering[i]
            val = clusterSizes[cluster] if closestPointToClusterMidpoint[cluster] == i else 0
            pointsToKeep.append(val)

        return pointsToKeep

    def calculateClustersMidpoint(self, points: list[list[float]], clustering: list[int]) -> list[list[float]]:
        numClusters = max(clustering) + 1
        clustersMidpoint = [[0, 0, 0]] * numClusters
        clustersQuantity = [0] * numClusters

        for i in range(len(points)):
            cluster = clustering[i]
            clustersMidpoint[cluster] = np.add(clustersMidpoint[cluster], points[i])
            clustersQuantity[cluster] += 1

        for cluster in range(max(clustering) + 1):
            clustersMidpoint[cluster] = np.divide(clustersMidpoint[cluster], clustersQuantity[cluster])

        return clustersMidpoint

    def calculateClosestPointToClusterMidpoint(self, points: list[list[float]], clustering: list[int], clustersMidpoint: list[list[float]]) -> list[int]:
        numClusters = len(clustersMidpoint)
        closestPointToClusterMidpoint = [-1] * numClusters
        closestDistanceToClusterMidpoint = [math.inf] * numClusters

        for i in range(len(points)):
            cluster = clustering[i]
            distance = math.dist(points[i], clustersMidpoint[cluster])
            if distance < closestDistanceToClusterMidpoint[cluster]:
                closestPointToClusterMidpoint[cluster] = i
                closestDistanceToClusterMidpoint[cluster] = distance

        return closestPointToClusterMidpoint

    def processFiles(self):
        coords = []
        countMaps = 0
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            countMaps += 1
            print("\treading " + filename)

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            for matchobj in re.finditer(Reducer._PATTERN, content):
                archetypeName = matchobj.group(2)
                scaling = [float(matchobj.group(6)), float(matchobj.group(6)), float(matchobj.group(8))]
                if not self.entityShouldBeConsidered(archetypeName, scaling):
                    continue

                coords.append([float(matchobj.group(3)), float(matchobj.group(4)), float(matchobj.group(5))])

        if not coords:
            return

        print("\treducing of " + str(countMaps) + " ymap files and in total " + str(len(coords)) + " entities")

        pointsToKeep = self.calculatePointsToKeep(coords)

        counter = [0]
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            content_new = re.sub(Reducer._PATTERN, lambda match: self.repl(match, pointsToKeep, counter), content)

            content_new = Ymap.replaceName(content_new, filename.lower()[:-9])
            content_new = Ymap.calculateAndReplaceLodDistance(content_new, self.ytypItems)
            content_new = Ymap.fixMapExtents(content_new, self.ytypItems)

            f = open(os.path.join(self.outputDir, filename.lower()), 'w')
            f.write(content_new)
            f.close()

    def repl(self, matchobj: Match, pointsToKeep: list[int], counter: list[int]) -> str:
        archetypeName = matchobj.group(2)
        scaling = [float(matchobj.group(6)), float(matchobj.group(6)), float(matchobj.group(8))]

        if not self.entityShouldBeConsidered(archetypeName, scaling):
            return matchobj.group(0)

        i = counter[0]
        counter[0] += 1
        if pointsToKeep[i] == 0:
            return ""
        elif not self.adaptScaling:
            return matchobj.group(0)

        scaleXY = math.pow(pointsToKeep[i], 2/5)
        scaleZ = math.pow(pointsToKeep[i], 2/5)

        bBoxSizes = self.ytypItems[archetypeName].boundingBox.getSizes()
        maxScalingXY = 40 / max(bBoxSizes[0], bBoxSizes[1])
        maxScalingZ = 60 / bBoxSizes[2]

        scaleXY = min(2.5, scaleXY, maxScalingXY)
        scaleZ = min(scaleXY, 2.5, scaleZ, maxScalingZ)  # scaleZ is at most scaleXY
        scaleXY = min(scaleXY, scaleZ * 1.25)  # scaleXY is at most 25 % larger than scaleZ

        scaleXY = scaling[0] * max(scaleXY, 1)  # ensure scaling does not decrease
        scaleZ = scaling[2] * max(scaleZ, 1)  # ensure scaling does not decrease

        return matchobj.group(1) + str(scaleXY) + matchobj.group(7) + str(scaleZ) + matchobj.group(9)

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.outputDir, lambda filename: not filename.endswith(".ymap.xml"))

    def entityShouldBeConsidered(self, archetypeName: str, scaling: list[float]):
        if archetypeName not in self.ytypItems:
            return False

        scaledRadius = self.ytypItems[archetypeName].boundingSphere.radius * max(scaling)

        return scaledRadius > 7
