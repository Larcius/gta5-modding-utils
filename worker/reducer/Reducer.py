import math
from re import Match
from natsort import natsorted
from typing import Optional
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

    defaultReducerResolution = 30.0

    defaultYmapPart: str
    ytypItems: dict[str, YtypItem]
    prefix: str
    reducerResolution: float
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

    groups = [
        ("prop_tree_pine_", "prop_tree_cedar_0", "prop_w_r_cedar_", "test_tree_cedar_trunk_001", "test_tree_forest_trunk_01", "prop_s_pine_dead_01", "prop_tree_fallen_pine_01", "prop_tree_birch_01", "prop_tree_birch_02", "prop_tree_birch_04", "prop_tree_jacada_", "prop_tree_lficus_", "prop_tree_oak_01", "prop_tree_olive_01", "prop_tree_eng_oak_01", "prop_tree_eucalip_01", "prop_bush_lrg_04"),
        ("prop_tree_birch_03", "prop_tree_maple_", "prop_tree_mquite_01", "prop_tree_stump_01", "test_tree_forest_trunk_base_01", "test_tree_forest_trunk_04", "prop_desert_iron_01", "prop_rio_del_01", "prop_rus_olive", "prop_rus_olive_wint"),
        ("prop_tree_cedar_s_", "prop_tree_cypress_01"),
        ("prop_bush_med_", "prop_bush_lrg_02", "prop_bush_lrg_03"),
        ("prop_cactus_", "prop_joshua_tree_"),
        ("prop_palm_", "prop_fan_palm_"),
        ("prop_rock_"),
        ("")  # everything else
    ]

    def __init__(self, inputDir: str, outputDir: str, prefix: str, reducerResolution: Optional[float], adaptScaling: bool):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix
        self.reducerResolution = reducerResolution if reducerResolution else self.defaultReducerResolution
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

        clustering, unused, unused = Util.performClusteringMaxFurthestDistance(points, self.reducerResolution)

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

        for cluster in range(numClusters):
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
        numGroups = len(self.groups)

        coords = []
        for group in range(numGroups):
            coords.append([])

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
                group = self.determineGroup(archetypeName, scaling)
                if group < 0:
                    continue

                coords[group].append([float(matchobj.group(3)), float(matchobj.group(4)), float(matchobj.group(5))])

        if not coords:
            return

        print("\treducing of " + str(countMaps) + " ymap files and in total " + str(len(coords)) + " entities")

        pointsToKeep = []
        for group in range(numGroups):
            pointsToKeep.append(self.calculatePointsToKeep(coords[group]))

        counter = [0] * numGroups
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            content_new = re.sub(Reducer._PATTERN, lambda match: self.repl(match, pointsToKeep, counter), content)

            content_new = Ymap.calculateAndReplaceLodDistance(content_new, self.ytypItems)
            content_new = Ymap.fixMapExtents(content_new, self.ytypItems)

            f = open(os.path.join(self.outputDir, filename.lower()), 'w')
            f.write(content_new)
            f.close()

    def repl(self, matchobj: Match, pointsToKeep: list[list[int]], counter: list[int]) -> str:
        archetypeName = matchobj.group(2)
        scaling = [float(matchobj.group(6)), float(matchobj.group(6)), float(matchobj.group(8))]

        group = self.determineGroup(archetypeName, scaling)
        if group < 0:
            return matchobj.group(0)

        i = counter[group]
        counter[group] += 1
        if pointsToKeep[group][i] == 0:
            return ""
        elif not self.adaptScaling or group != 0:
            return matchobj.group(0)

        # TODO consider scaleZ for position update (depending on rotation and offsetZ; see z-fixer)
        # TODO take into account the total area divided by the area of this entity
        scaleXY = math.pow(pointsToKeep[group][i], 2/5)
        scaleZ = math.pow(pointsToKeep[group][i], 2/5)

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

    def determineGroup(self, archetypeName: str, scaling: list[float]) -> int:
        for group in range(len(self.groups)):
            if archetypeName.startswith(self.groups[group]):
                return group

        return -1
