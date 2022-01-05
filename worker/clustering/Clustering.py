import numpy as np
from natsort import natsorted
from matplotlib import pyplot
import math
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

    MAX_EXTEND = 1500

    _PATTERN = re.compile(
        '[\t ]*<Item type="CEntityDef">' +
        '\\s*<archetypeName>[^<]+</archetypeName>' +
        '(?:\\s*<[^/].*>)*' +
        '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
        '(?:\\s*<[^/].*>)*' +
        '\\s*</Item>[\r\n]+'
    )

    def __init__(self, inputDir: str, outputDir: str, prefix: str, numCluster: int):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix
        self.numCluster = numCluster

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

    def processFiles(self):
        coords = []
        customPostEntities = []
        numFiles = 0
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            print("\treading " + filename)

            numFiles += 1

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

        print("\tperforming clustering of " + str(numFiles) + " ymap files and in total " + str(len(coords)) + " entities")

        if self.numCluster > 0:
            clusters, unused, furthestDistances = Util.performClusteringFixedNumClusters(coords, self.numCluster)
        else:
            clusters, furthestDistances = Util.performClustering(coords, -1, Clustering.MAX_EXTEND)

        numClusters = len(np.unique(clusters))

        numDigitsMapIndices = math.ceil(math.log(numClusters, 10))

        outputFiles = {}
        for cluster in np.unique(clusters):
            outputFiles[cluster] = open(os.path.join(self.outputDir, self.prefix + "_" + str(cluster + 1).zfill(numDigitsMapIndices) + ".ymap.xml"), 'w')
            outputFiles[cluster].write(contentPreEntities)

        i = 0
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()
            f.close()

            for matchobj in re.finditer(Clustering._PATTERN, content):
                cluster = clusters[i]
                outputFiles[cluster].write(matchobj.group(0))
                i += 1

        for cluster in np.unique(clusters):
            outputFiles[cluster].write(contentPostEntities)
            outputFiles[cluster].close()

        for custom in customPostEntities:
            print("custom content after </entities> in file " + custom)

        self.plotClusterResult(coords, clusters)

    def plotClusterResult(self, coords: list[list[float]], clusters):
        # create scatter plot for samples from each cluster
        cmap = pyplot.cm.get_cmap("gist_ncar", len(np.unique(clusters)) + 1)
        X = np.array(coords)
        i = 0
        for cluster in np.unique(clusters):
            # get row indexes for samples with this cluster
            row_ix = np.where(clusters == cluster)

            # create scatter of these samples
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], color=cmap(i))

            i += 1

        pyplot.gca().set_aspect('equal')

        # show the plot
        pyplot.show()

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
