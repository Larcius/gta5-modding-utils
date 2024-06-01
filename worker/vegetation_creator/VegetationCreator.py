import math

import numpy as np
import os
import random
import re

from numpy import ndarray
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as pyplot

from natsort import natsorted

from common.Util import Util


class VegetationCreator:
    GROUPS = [
        [
            {
                "prop_tree_maple_02",
                "prop_tree_maple_03",
                "prop_tree_mquite_01",
                "prop_desert_iron_01",
                "prop_rus_olive_wint",
                "prop_rio_del_01"
            }, {
                "prop_tree_birch_01",
                "prop_tree_birch_02",
                "prop_tree_birch_03",
                "prop_tree_birch_03b",
                "prop_tree_birch_04"
            }, {
                "prop_tree_cedar_02",
                "prop_tree_cedar_03",
                "prop_tree_cedar_04",
                "prop_tree_cedar_s_01",
                "prop_tree_cedar_s_02",
                "prop_tree_pine_01",
                "prop_tree_pine_02",
                "prop_w_r_cedar_01",
                "test_tree_cedar_trunk_001",
                "test_tree_forest_trunk_01",
                "test_tree_forest_trunk_04",
                "prop_w_r_cedar_dead",
                "prop_tree_fallen_pine_01",
                "prop_s_pine_dead_01"
            }, {
                "prop_tree_cedar_s_04",
                "prop_tree_cedar_s_05",
                "prop_tree_cedar_s_06",
                "prop_tree_cypress_01"
            }, {
                "prop_tree_oak_01",
                "prop_tree_eng_oak_01",
                "prop_rus_olive",
                "prop_tree_olive_01",
                "prop_tree_eucalip_01",
                "prop_bush_lrg_04b",
                "prop_bush_lrg_04c",
                "prop_bush_lrg_04d"
            }, {
                "prop_tree_lficus_02",
                "prop_tree_lficus_03",
                "prop_tree_lficus_05",
                "prop_tree_lficus_06"
            }, {
                "prop_tree_jacada_01",
                "prop_tree_jacada_02"
            }, {
                "prop_tree_stump_01",
                "test_tree_forest_trunk_base_01"
            }, {
                "prop_tree_birch_05",
                "prop_veg_crop_orange"
            }
        ], [
            {
                "prop_fan_palm_01a",
                "prop_palm_fan_02_a",
                "prop_palm_fan_02_b",
                "prop_palm_fan_03_a",
                "prop_palm_fan_03_b",
                "prop_palm_fan_03_c",
                "prop_palm_fan_03_c_graff",
                "prop_palm_fan_03_d",
                "prop_palm_fan_03_d_graff",
                "prop_palm_fan_04_a",
                "prop_palm_fan_04_b",
                "prop_palm_fan_04_c",
                "prop_palm_fan_04_d",
                "prop_palm_huge_01a",
                "prop_palm_huge_01b",
            }, {
                "prop_palm_sm_01a",
                "prop_palm_sm_01d",
                "prop_palm_sm_01e",
                "prop_palm_sm_01f",
                "prop_palm_med_01a",
                "prop_palm_med_01b",
                "prop_palm_med_01c",
                "prop_palm_med_01d",
            }
        ], [
            {
                "prop_cactus_01a",
                "prop_cactus_01b",
                "prop_cactus_01c",
                "prop_cactus_01d",
                "prop_cactus_01e"
            }, {
                "prop_joshua_tree_01a",
                "prop_joshua_tree_01b",
                "prop_joshua_tree_01c",
                "prop_joshua_tree_01d",
                "prop_joshua_tree_01e",
                "prop_joshua_tree_02a",
                "prop_joshua_tree_02b",
                "prop_joshua_tree_02c",
                "prop_joshua_tree_02d",
                "prop_joshua_tree_02e"
            }
        ]
    ]

    # TODO not yet used
    GROUPS_SMALL = [
        {
            "prop_bush_med_01",
            "prop_bush_med_02",
            "prop_bush_med_03",
            "prop_bush_med_05",
            "prop_bush_med_06",
            "prop_bush_med_07"
        }, {
            "prop_bush_lrg_01",
            "prop_bush_lrg_01b",
            "prop_bush_lrg_01c",
            "prop_bush_lrg_01d",
            "prop_bush_lrg_01e",
            "prop_bush_lrg_03",
            "prop_am_box_wood_01",
            "prop_bush_lrg_02",
            "prop_bush_lrg_02b"
        }, {
            "prop_veg_crop_tr_01",
            "prop_veg_crop_tr_02",
            "prop_agave_01",
            "prop_agave_02",
            "prop_aloevera_01",
            "prop_bush_dead_02",
            "prop_cat_tail_01",
            "prop_p_spider_01a",
            "prop_p_spider_01c",
            "prop_p_spider_01d",
            "prop_plant_01a",
            "prop_plant_cane_01b",
            "prop_plant_cane_02a",
            "prop_plant_cane_02b",
            "prop_plant_fern_02a",
            "prop_plant_fern_02b",
            "prop_plant_palm_01a",
            "prop_plant_palm_01b",
            "prop_plant_palm_01c",
            "prop_cactus_02",
            "prop_cactus_03"
        }, {
            "prop_plant_01b",
            "prop_plant_cane_01a",
            "prop_plant_fern_01a",
            "prop_plant_fern_01b",
            "prop_plant_paradise",
            "prop_plant_paradise_b"
        }, {
            "prop_rock_1_a",
            "prop_rock_1_b",
            "prop_rock_1_c",
            "prop_rock_1_d",
            "prop_rock_1_e",
            "prop_rock_1_f",
            "prop_rock_1_g",
            "prop_rock_1_h",
            "prop_rock_1_i"
        }, {
            "prop_rock_2_a",
            "prop_rock_2_c",
            "prop_rock_2_d",
            "prop_rock_2_f",
            "prop_rock_2_g"
        }, {
            "prop_rock_3_a",
            "prop_rock_3_b",
            "prop_rock_3_c",
            "prop_rock_3_d",
            "prop_rock_3_e",
            "prop_rock_3_f",
            "prop_rock_3_g",
            "prop_rock_3_h",
            "prop_rock_3_i",
            "prop_rock_3_j"
        }, {
            "prop_rock_4_a",
            "prop_rock_4_b",
            "prop_rock_4_c",
            "prop_rock_4_d",
            "prop_rock_4_e",
            "prop_rock_4_big",
            "prop_rock_4_big2",
            "prop_rock_4_cl_1",
            "prop_rock_4_cl_2",
            "prop_rock_5_a",
            "prop_rock_5_b",
            "prop_rock_5_c",
            "prop_rock_5_d",
            "prop_rock_5_e",
            "prop_rock_5_smash1",
            "prop_rock_5_smash3",
            "rock_4_cl_2_1",
            "rock_4_cl_2_2"
        }
    ]

    CANDIDATES_LOW_PROBABILITY = {
        "prop_tree_cedar_s_01",
        "prop_tree_cedar_s_02"
    }

    CANDIDATES_VERY_LOW_PROBABILITY = {
        "prop_w_r_cedar_dead",
        "prop_tree_fallen_pine_01",
        "prop_s_pine_dead_01"
    }

    EXCLUDE_CANDIDATES = {
        "test_tree_forest_trunk_01",
        "test_tree_forest_trunk_04",
        "prop_palm_fan_03_c_graff",
        "prop_palm_fan_03_d_graff"
    }

    # init archetypeGroupMapping
    ARCHETYPE_SUPERGROUP_MAPPING = dict()
    ARCHETYPE_GROUP_MAPPING = dict()
    for supergroup in GROUPS:
        for group in supergroup:
            for archetype in group:
                ARCHETYPE_SUPERGROUP_MAPPING[archetype] = supergroup
                ARCHETYPE_GROUP_MAPPING[archetype] = group

    TRIANGLE_DISTANCE_MIN = 12
    TRIANGLE_DISTANCE_MAX = 60
    TRIANGLE_MIN_ANGLE = math.pi / 12

    MAP_NAME_SUFFIX = "_vegetation_creator"


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
        self.copyInput()
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
        mapNames = []
        points = []
        archetypes = []
        for filename in natsorted(os.listdir(self.inputDir)):
            if filename.endswith(".ymap.xml"):
                mapNames.append(Util.getMapnameFromFilename(filename))
                self.processFile(filename, points, archetypes)

        countInitPoints = len(points)
        if countInitPoints == 0:
            return

        origPoints2d = np.array(points)[:, :2]

        pointsAdded = True
        while pointsAdded:
            pointsAdded = False

            points2d = np.array(points)[:, :2]
            tri = Delaunay(points2d)

            for simplex in tri.simplices:
                simplexArchetypes = [archetypes[simplex[0]], archetypes[simplex[1]], archetypes[simplex[2]]]
                simplexVertices = [points[simplex[0]], points[simplex[1]], points[simplex[2]]]

                simplexVertices2d = [points2d[simplex[0]], points2d[simplex[1]], points2d[simplex[2]]]
                pairwiseDistances2d = pdist(simplexVertices2d)
                if min(pairwiseDistances2d) < VegetationCreator.TRIANGLE_DISTANCE_MIN or max(pairwiseDistances2d) > VegetationCreator.TRIANGLE_DISTANCE_MAX:
                    continue

                if not VegetationCreator.shareSameSuperGroup(simplexArchetypes):
                    continue

                minAngle = math.pi
                for j in range(3):
                    minAngle = min(minAngle, abs(Util.calculateAngle(simplexVertices2d[j], simplexVertices2d[(j + 1) % 3], simplexVertices2d[(j + 2) % 3])))

                if minAngle < VegetationCreator.TRIANGLE_MIN_ANGLE:
                    continue

                factors = [random.uniform(0.4, 1), random.uniform(0.4, 1), random.uniform(0.4, 1)]
                factors = np.divide(factors, sum(factors))

                center = np.zeros(3)
                for j in range(3):
                    center += np.multiply(simplexVertices[j], factors[j])

                points.append(center)
                archetypes.append(random.choice(simplexArchetypes))
                pointsAdded = True

        newMapName = self.getNewMapName(mapNames)
        newArchetypes = self.computeArchetypeNames(points, archetypes[0:countInitPoints])[countInitPoints:]
        newPoints = points[countInitPoints:]
        self.createYmap(newMapName, newPoints, newArchetypes)

        pyplot.gca().set_aspect('equal')
        pyplot.plot(np.array(newPoints)[:, 0], np.array(newPoints)[:, 1], 'ro')
        pyplot.plot(origPoints2d[:, 0], origPoints2d[:, 1], 'go')
        pyplot.show()

    @staticmethod
    def shareSameSuperGroup(archetypes: list[str]) -> bool:
        prevSuperGroup = None
        for archetype in archetypes:
            superGroup = VegetationCreator.ARCHETYPE_SUPERGROUP_MAPPING.get(archetype)
            if prevSuperGroup is None:
                prevSuperGroup = superGroup
            elif prevSuperGroup != superGroup:
                return False
        return True

    @staticmethod
    def computeArchetypeNames(points: list[ndarray], archetypes: list[str]) -> list[str]:
        points2d = np.array(points)[:, :2]
        kdTree = KDTree(points2d)
        todo = list(range(len(archetypes), len(points2d)))
        nthClosest = {}
        result = {i: archetypes[i] for i in range(len(archetypes))}
        while len(todo) > 0:
            i = todo.pop(0)
            nthClosestCur = nthClosest.get(i)
            if nthClosestCur is None:
                nthClosestCur = 1

            point2d = points2d[i]
            nearestNeighborIndex = kdTree.query(point2d, nthClosestCur + 1)[1][nthClosestCur]
            if nearestNeighborIndex in result:
                result[i] = VegetationCreator.getRandomArchetypeWithinGroup(result[nearestNeighborIndex])
                if i in nthClosest:
                    del nthClosest[i]
                continue

            if nearestNeighborIndex in nthClosest:
                nthClosest[i] = nthClosestCur + 1
                todo.append(i)
            else:
                nthClosest[i] = nthClosestCur
                todo.remove(nearestNeighborIndex)
                todo.insert(0, nearestNeighborIndex)
                todo.append(i)

        return [result[key] for key in sorted(result.keys())]

    @staticmethod
    def getRandomArchetypeWithinGroup(archetype: str) -> str:
        group = VegetationCreator.ARCHETYPE_GROUP_MAPPING.get(archetype)
        if group is None:
            return archetype

        props = group.difference(VegetationCreator.EXCLUDE_CANDIDATES)
        if random.random() > 0.1:
            props = props.difference(VegetationCreator.CANDIDATES_VERY_LOW_PROBABILITY)
        if random.random() > 0.4:
            props = props.difference(VegetationCreator.CANDIDATES_LOW_PROBABILITY)

        # need to be sorted to get reproducible results in random.choice() for a fixed seed
        choices = sorted(list(props))

        return random.choice(choices)

    def createYmap(self, mapName: str, points: list[list[float]], archetypes: list[str]):
        contentEntities = ""
        for i in range(len(points)):
            point = points[i]
            archetype = archetypes[i]

            contentEntities += self.contentTemplateEntity \
                .replace("${NAME}", archetype) \
                .replace("${POSITION.X}", Util.floatToStr(point[0])) \
                .replace("${POSITION.Y}", Util.floatToStr(point[1])) \
                .replace("${POSITION.Z}", Util.floatToStr(point[2]))

        map = self.contentTemplateMap \
            .replace("${NAME}", mapName) \
            .replace("${ENTITIES}\n", contentEntities)

        fileMap = open(os.path.join(self.outputDir, Util.getFilenameFromMapname(mapName)), 'w')
        fileMap.write(map)
        fileMap.close()

    def processFile(self, filename: str, points: list[list[float]], archetypes: list[str]):
        print("\tprocessing " + filename)

        f = open(os.path.join(self.inputDir, filename), 'r')
        content = f.read()
        f.close()

        for match in re.finditer('<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*</Item>', content):

            archetypeName = match.group(1)
            if archetypeName not in VegetationCreator.ARCHETYPE_GROUP_MAPPING:
                continue

            points.append([float(match.group(2)), float(match.group(3)), float(match.group(4))])
            archetypes.append(archetypeName)

    def getNewMapName(self, mapNames: list[str]):
        prefixes = Util.determinePrefixBundles(mapNames)
        if len(prefixes) == 1:
            mapName = prefixes[0].rstrip("_") + VegetationCreator.MAP_NAME_SUFFIX
        else:
            mapName = self.prefix.rstrip("_") + VegetationCreator.MAP_NAME_SUFFIX

        idx = 1
        finalMapName = mapName
        while finalMapName in mapNames:
            finalMapName = mapName + "_" + str(idx)
            idx += 1

        return finalMapName

    def copyInput(self):
        Util.copyFiles(self.inputDir, self.outputDir)
