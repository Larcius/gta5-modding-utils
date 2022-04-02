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

    defaultPostEntities: str
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
        self.readTemplateDefaultPostEntities()
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

    def readTemplateDefaultPostEntities(self):
        f = open(os.path.join(os.path.dirname(__file__), "default_post_entities.xml"), 'r')
        self.defaultPostEntities = f.read()
        f.close()

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

    def getNumGroupsAndNumClusters(self, hierarchy: list[list[int]]) -> (int, int):
        numGroups = 0
        numClusters = {}
        clustersCounted = {}
        for h in hierarchy:
            cluster = h[0]
            group = h[1]
            if group not in numClusters:
                clustersCounted[group] = set()
                numClusters[group] = 0
                numGroups += 1

            if cluster not in clustersCounted[group]:
                clustersCounted[group].add(cluster)
                numClusters[group] += 1

        return numGroups, numClusters

    def processFiles(self):
        coords = []
        customPostEntities = []
        mapNames = []
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            print("\treading " + filename)

            mapNames.append(filename[:-9])

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            indexEntitiesStart = content.find("  <entities>")
            if indexEntitiesStart < 0:
                continue
            indexEntitiesEnd = content.index("  </entities>")

            contentPreEntities = content[:indexEntitiesStart + 13]
            contentPostEntities = content[indexEntitiesEnd:]

            if not contentPostEntities.startswith(self.defaultPostEntities):
                customPostEntities.append(filename)

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

        numGroups, numClusters = self.getNumGroupsAndNumClusters(hierarchy)

        outputFiles = {}
        mapPrefix = self.getMapPrefix(mapNames)
        for h in hierarchy:
            cluster = h[0]
            group = h[1]
            if group not in outputFiles:
                outputFiles[group] = {}

            if cluster not in outputFiles[group]:
                clusterName = self.getClusterName(group, cluster, numGroups, numClusters[group])
                outputFiles[group][cluster] = open(os.path.join(self.outputDir, mapPrefix + ("_" if clusterName else "") + clusterName + ".ymap.xml"), 'w')
                outputFiles[group][cluster].write(contentPreEntities)

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
                outputFiles[group][cluster].write(matchobj.group(0))
                i += 1

        for g in outputFiles:
            groups = outputFiles[g]
            for c in groups:
                file = groups[c]
                file.write(contentPostEntities)
                file.close()

        for custom in customPostEntities:
            print("custom content after </entities> in file " + custom)

        self.plotClusterResult(coords, hierarchy)

    def plotClusterResult(self, coords: list[list[float]], hierarchy: list[list[int]]):
        numGroups, numClusters = self.getNumGroupsAndNumClusters(hierarchy)
        numTotalClusters = 0
        for c in numClusters:
            numTotalClusters += numClusters[c]

        groups = {}
        i = 0
        for h in hierarchy:
            cluster = h[0]
            group = h[1]
            if group not in groups:
                groups[group] = {}

            if cluster not in groups[group]:
                groups[group][cluster] = []

            groups[group][cluster].append(i)
            i += 1

        # create scatter plot for samples from each cluster
        cmap = pyplot.cm.get_cmap("gist_ncar", numTotalClusters + 1)
        X = np.array(coords)
        i = 0
        for group in groups:
            for cluster in groups[group]:
                # get row indexes for samples with this cluster
                row_ix = groups[group][cluster]

                # create scatter of these samples
                pyplot.scatter(X[row_ix, 0], X[row_ix, 1], color=cmap(i))
                clusterName = self.getClusterName(group, cluster, numGroups, numClusters[group])
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
