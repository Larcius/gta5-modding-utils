import math
from typing import Optional
import numpy as np
from natsort import natsorted
from matplotlib import pyplot
import matplotlib.patheffects as PathEffects
import os
import re

from common.Util import Util
from common.ymap.Ymap import Ymap
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class Clustering:
    inputDir: str
    outputDir: str

    defaultYmapPart: str
    ymapTemplate: str
    ytypItems: dict[str, YtypItem]
    prefix: str
    numCluster: int

    GROUP_MAX_EXTEND = 2000
    MAX_EXTEND = 600

    _PATTERN = re.compile(
        '[\t ]*<Item type="CEntityDef">' +
        '\\s*<archetypeName>[^<]+</archetypeName>' +
        '(?:\\s*<[^/].*>)*?' +
        '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
        '(?:\\s*<[^/].*>)*?' +
        '\\s*</Item>[\r\n]+'
    )

    def __init__(self, inputDir: str, outputDir: str, prefix: str, numCluster: int, clusteringPrefix: Optional[str]):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix
        self.numCluster = numCluster
        self.clusteringPrefix = clusteringPrefix

    def run(self):
        print("running clustering...")
        self.readYtyps()
        self.readYmapTemplate()
        self.createOutputDir()
        self.processFiles()
        self.fixMapExtents()
        self.copyOthers()
        print("clustering DONE")

    def readYtyps(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "../..", "resources", "ytyp"))

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)

    def readYmapTemplate(self):
        f = open(os.path.join(os.path.dirname(__file__), "templates", "template.ymap.xml"), 'r')
        self.ymapTemplate = f.read()
        f.close()

        self.defaultYmapPart = self.getYmapPartAfterEntitiesAndBeforeBlock(self.ymapTemplate)
        if self.defaultYmapPart is None or not self.defaultYmapPart:
            raise Exception("invalid ymap template")

    def getYmapPartAfterEntitiesAndBeforeBlock(self, ymap: str):
        startIndex = ymap.find("\n  </entities>")
        if startIndex < 0:
            startIndex = ymap.find("\n  <entities/>")

        if startIndex < 0:
            return None

        endIndex = ymap.rfind("\n  <block>")
        if endIndex < 0:
            endIndex = ymap.rfind("\n  <block/>")

        if endIndex < 0:
            return None

        return ymap[startIndex + 14:endIndex]

    def _calculateMapHierarchy(self, points: list[list[float]], hierarchy: list[list[int]]) -> list[list[int]]:
        level = len(hierarchy[0])
        if level == 0:
            maxExtends = Clustering.GROUP_MAX_EXTEND
            unevenClusters = False
        elif level == 1:
            maxExtends = Clustering.MAX_EXTEND
            unevenClusters = False
        else:
            return hierarchy

        absIndices = []
        pointsOfParent = []
        for i in range(len(points)):
            parentIndex = 0 if len(hierarchy[i]) == 0 else hierarchy[i][0]

            while parentIndex >= len(pointsOfParent):
                absIndices.append([])
                pointsOfParent.append([])

            absIndices[parentIndex].append(i)
            pointsOfParent[parentIndex].append(points[i])

        for parentIndex in range(len(pointsOfParent)):
            clustering, unused = Util.performClustering(pointsOfParent[parentIndex], -1, maxExtends, unevenClusters)

            for c in range(len(clustering)):
                i = absIndices[parentIndex][c]
                hierarchy[i].insert(0, clustering[c])

        return self._calculateMapHierarchy(points, hierarchy)

    def calculateMapHierarchy(self, points: list[list[float]]) -> list[list[int]]:
        if len(points) == 0:
            return []

        hierarchy = []
        for i in range(len(points)):
            hierarchy.append([])

        return self._calculateMapHierarchy(points, hierarchy)

    def getClusterName(self, group: int, cluster: int, numGroups: int, numClusters: int) -> str:
        letters = ""
        if numGroups > 1:
            numLetters = math.ceil(math.log(numGroups, 26))
            n = group
            while n > 0:
                letters = chr(97 + (n % 26)) + letters
                n = math.floor(n / 26)
            letters = letters.rjust(numLetters, "a")

        if numClusters > 1:
            numDigits = math.ceil(math.log(numClusters, 10))
            digits = str(cluster).zfill(numDigits)
        else:
            digits = ""

        return letters + ("_" if letters and digits else "") + digits

    def processFiles(self):
        coords = []
        mapsHavingNotOnlyEntities = []
        mapNames = []
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            print("\treading " + filename)

            mapName = filename[:-9]
            mapNames.append(mapName)

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            ymapPartAfterEntitiesAndBeforeBlock = self.getYmapPartAfterEntitiesAndBeforeBlock(content)
            if ymapPartAfterEntitiesAndBeforeBlock != self.defaultYmapPart:
                mapsHavingNotOnlyEntities.append(mapName)

            for matchobj in re.finditer(Clustering._PATTERN, content):
                coords.append([float(matchobj.group(1)), float(matchobj.group(2)), float(matchobj.group(3))])

        if not coords:
            return

        print("\tperforming clustering of " + str(len(mapNames)) + " ymap files and in total " + str(len(coords)) + " entities")

        if self.numCluster > 0:
            clusters, unused, furthestDistances = Util.performClusteringFixedNumClusters(coords, self.numCluster)
            hierarchy = [[i, 0] for i in clusters]
        else:
            hierarchy = self.calculateMapHierarchy(coords)

        outputFiles = {}
        mapPrefix = self.getMapPrefix(mapNames)
        for h in hierarchy:
            cluster = h[0]
            group = h[1]
            if group not in outputFiles:
                outputFiles[group] = {}

            if cluster not in outputFiles[group]:
                outputFiles[group][cluster] = ""

        i = 0
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            for matchobj in re.finditer(Clustering._PATTERN, content):
                cluster = hierarchy[i][0]
                group = hierarchy[i][1]
                outputFiles[group][cluster] += matchobj.group(0)
                i += 1

        self.writeClusteredYmap(mapPrefix, outputFiles)

        for mapName in mapsHavingNotOnlyEntities:
            content = Util.readFile(os.path.join(self.inputDir, mapName + ".ymap.xml"))

            newMapName = mapName
            if os.path.exists(os.path.join(self.outputDir, newMapName + ".ymap.xml")):
                newMapName = re.sub("_no_entities\\d*$", "", newMapName) + "_no_entities"
                i = -1
                while os.path.exists(os.path.join(self.outputDir, newMapName + ("" if i < 0 else str(i)) + ".ymap.xml")):
                    i += 1
                if i >= 0:
                    newMapName += str(i)

            content = Ymap.replaceParent(content, None)
            content = Ymap.replaceName(content, newMapName)
            content = re.sub("<entities>[\\S\\s]*</entities>", "<entities/>", content)

            Util.writeFile(os.path.join(self.outputDir, newMapName + ".ymap.xml"), content)

        self.plotClusterResult(coords, hierarchy)

    def writeClusteredYmap(self, mapPrefix: str, clusteredEntities: dict[int, dict[int, str]]):
        numGroups = len(clusteredEntities)
        for group in clusteredEntities:
            clustersInGroup = clusteredEntities[group]
            numClusters = len(clustersInGroup)
            for cluster in clustersInGroup:
                clusterName = self.getClusterName(group, cluster, numGroups, numClusters)
                mapName = mapPrefix + ("_" if clusterName else "") + clusterName

                entities = clustersInGroup[cluster]
                ymapContent = self.createYmapContent(mapName, entities)

                file = open(os.path.join(self.outputDir, mapName + ".ymap.xml"), 'w')
                file.write(ymapContent)
                file.close()

    def createYmapContent(self, mapName: str, entities: str) -> str:
        return self.ymapTemplate \
            .replace("${NAME}", mapName) \
            .replace("${TIMESTAMP}", Util.getNowInIsoFormat()) \
            .replace("${ENTITIES}\n", entities)


    def plotClusterResult(self, coords: list[list[float]], hierarchy: list[list[int]]):
        numTotalClusters = 0
        groups = {}
        i = 0
        for h in hierarchy:
            cluster = h[0]
            group = h[1]
            if group not in groups:
                groups[group] = {}

            if cluster not in groups[group]:
                groups[group][cluster] = []
                numTotalClusters += 1

            groups[group][cluster].append(i)
            i += 1

        numGroups = len(groups)

        # create scatter plot for samples from each cluster
        cmap = pyplot.cm.get_cmap("gist_ncar", numTotalClusters + 1)
        X = np.array(coords)
        i = 0
        for group in groups:
            numClusters = len(groups[group])
            for cluster in groups[group]:
                # get row indexes for samples with this cluster
                row_ix = groups[group][cluster]

                # create scatter of these samples
                pyplot.scatter(X[row_ix, 0], X[row_ix, 1], color=cmap(i))
                clusterName = self.getClusterName(group, cluster, numGroups, numClusters)
                annotate = pyplot.annotate(clusterName, xy=(np.mean(X[row_ix, 0]), np.mean(X[row_ix, 1])), ha='center', va='center')
                annotate.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='w')])

                i += 1

        pyplot.gca().set_aspect('equal')

        pyplot.show(block=False)

    def getMapPrefix(self, mapNames: list[str]):
        if self.clusteringPrefix is not None:
            return self.clusteringPrefix

        prefixes = Util.determinePrefixBundles(mapNames)
        if len(prefixes) == 1:
            return prefixes[0]
        else:
            return self.prefix

    # adapt extents and set current datetime
    def fixMapExtents(self):
        print("\tfixing map extents")

        for filename in natsorted(os.listdir(self.outputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            file = open(os.path.join(self.outputDir, filename), 'r')
            content = file.read()
            file.close()

            content = Ymap.replaceName(content, filename.lower()[:-9])
            content = Ymap.replaceParent(content, None)
            content = Ymap.fixMapExtents(content, self.ytypItems)

            file = open(os.path.join(self.outputDir, filename), 'w')
            file.write(content)
            file.close()

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.outputDir, lambda filename: not filename.endswith(".ymap.xml"))
