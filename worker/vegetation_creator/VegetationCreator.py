import math

import numpy as np
import os
import random
import re
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
import matplotlib.pyplot as pyplot

from natsort import natsorted

from common.Util import Util


class VegetationCreator:
    CANDIDATES = {
        #"prop_tree_mquite_01",
        #"prop_tree_birch_01",
        #"prop_tree_birch_02",
        #"prop_tree_birch_03",
        #"prop_tree_birch_03b",
        #"prop_tree_birch_04",
        "prop_tree_cedar_02",
        "prop_tree_cedar_03",
        "prop_tree_cedar_04",
        #"prop_tree_cedar_s_01",
        #"prop_tree_cedar_s_02",
        #"prop_tree_eng_oak_01",
        #"prop_tree_maple_02",
        #"prop_tree_maple_03",
        #"prop_tree_oak_01",
        "prop_tree_pine_01",
        "prop_tree_pine_02",
        #"prop_tree_stump_01",
        "prop_w_r_cedar_01",
        "test_tree_cedar_trunk_001",
        #"prop_bush_lrg_04b",
        #"prop_bush_lrg_04c",
        #"prop_bush_lrg_04d",
    }

    CANDIDATES_LOW_PROBABILITY = {
        "prop_w_r_cedar_dead",
        "prop_tree_fallen_pine_01",
        "prop_s_pine_dead_01",
    }

    TRIANGLE_MIN_ANGLE = math.pi / 8

    MAP_NAME = "vegetation_creator"


    inputDir: str
    outputDir: str
    prefix: str
    contentTemplateEntity: str
    contentTemplateMap: str


    def __init__(self, inputDir: str, outputDir: str, prefix: str):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix

        # using a specific seed to be able to get reproducible results
        random.seed(a=0)

    def run(self):
        print("running vegetation creator...")
        self.readTemplates()
        self.createOutputDir()
        self.processFiles()
        print("vegetation creator DONE")

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)


    def readTemplates(self):
        templatesDir = os.path.join(os.path.dirname(__file__), "templates")

        f = open(os.path.join(templatesDir, "template_entity.ymap.xml"), 'r')
        self.contentTemplateEntity = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template.ymap.xml"), 'r')
        self.contentTemplateMap = f.read()
        f.close()


    def processFiles(self):
        points = []
        for filename in natsorted(os.listdir(self.inputDir)):
            if filename.endswith(".ymap.xml"):
                self.processFile(filename, points)

        if len(points) == 0:
            return

        origPoints2d = np.array(points)[:, :2]

        newPoints = []
        for i in range(2):
            points2d = np.array(points)[:, :2]
            tri = Delaunay(points2d)

            for simplex in tri.simplices:
                simplexVertices = [points[simplex[0]], points[simplex[1]], points[simplex[2]]]

                pairwiseDistances = pdist(simplexVertices)

                if min(pairwiseDistances) < 16 or max(pairwiseDistances) > 60:
                    continue

                minAngle = math.pi
                for j in range(3):
                    minAngle = min(minAngle, abs(Util.calculateAngle(simplexVertices[j], simplexVertices[(j + 1) % 3], simplexVertices[(j + 2) % 3])))

                if minAngle < VegetationCreator.TRIANGLE_MIN_ANGLE:
                    continue

                factors = [random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)]
                factors = np.divide(factors, sum(factors))

                center = np.zeros(3)
                for j in range(3):
                    center += np.multiply(simplexVertices[j], factors[j])

                newPoints.append(center)
                points.append(center)

        self.createYmap(newPoints)

        pyplot.gca().set_aspect('equal')
        pyplot.plot(np.array(newPoints)[:, 0], np.array(newPoints)[:, 1], 'ro')
        pyplot.plot(origPoints2d[:, 0], origPoints2d[:, 1], 'go')
        pyplot.show()


    def createYmap(self, points: list[list[float]]):
        contentEntities = ""
        for point in points:
            if random.random() < 0.1:
                props = set().union(VegetationCreator.CANDIDATES).union(VegetationCreator.CANDIDATES_LOW_PROBABILITY)
            else:
                props = VegetationCreator.CANDIDATES

            # need to be sorted to get reproducible results in random.choice() for a fixed seed
            choices = sorted(list(props))

            archetype = random.choice(choices)

            contentEntities += self.contentTemplateEntity \
                .replace("${NAME}", archetype) \
                .replace("${POSITION.X}", Util.floatToStr(point[0])) \
                .replace("${POSITION.Y}", Util.floatToStr(point[1])) \
                .replace("${POSITION.Z}", Util.floatToStr(point[2]))

        map = self.contentTemplateMap \
            .replace("${NAME}", VegetationCreator.MAP_NAME) \
            .replace("${ENTITIES}\n", contentEntities)

        fileMap = open(os.path.join(self.outputDir, VegetationCreator.MAP_NAME + ".ymap.xml"), 'w')
        fileMap.write(map)
        fileMap.close()


    def processFile(self, filename: str, points: list[list[float]]):
        print("\tprocessing " + filename)

        f = open(os.path.join(self.inputDir, filename), 'r')
        content = f.read()
        f.close()

        for match in re.finditer('<Item type="CEntityDef">' +
                             '\\s*<archetypeName>(?:prop_tree|prop_s_pine|prop_w_r_cedar|test_tree)[^<]*</archetypeName>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*</Item>', content):

            points.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
