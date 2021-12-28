# TODO refactoring needed

import os
import re
import random
import shutil
import math
from dataclasses import dataclass
from natsort import natsorted


@dataclass
class Tree:
    height: float
    onlyFlatlands: bool
    trunkRadius: float
    offsetZ: float = 0

    def getLodDistance(self, scaleZ):
        return math.sqrt(self.height * scaleZ) * 60


# using a specific seed to be able to get reproducible results
random.seed(a=0)

# trees can be found at x64i.rpf\levels\gta5\props\vegetation\v_trees.rpf\
# and update\x64\dlcpacks\patchday2ng\dlc.rpf\x64\levels\gta5\props\vegetation\v_trees.rpf

trees = {
    "Prop_S_Pine_Dead_01": Tree(12, False, 0.55),
    "Prop_Tree_Birch_01": Tree(16, True, 0.95),
    "Prop_Tree_Birch_02": Tree(12, True, 0.63),
    "Prop_Tree_Birch_03": Tree(8, True, 0.24, -0.08),
    "Prop_Tree_Birch_03b": Tree(6, True, 0.18),
    "Prop_Tree_Birch_04": Tree(13, True, 0.57),
    "Prop_Tree_Cedar_02": Tree(31, False, 0.89),
    "Prop_Tree_Cedar_03": Tree(36, False, 0.65),
    "Prop_Tree_Cedar_04": Tree(33, False, 1.1),
    "Prop_Tree_Cedar_S_01": Tree(10, True, 0.52),
    "Prop_Tree_Cedar_S_02": Tree(5, True, 0.13),
    # "Prop_Tree_Cedar_S_04": Tree(17, True, 1),
    # "Prop_Tree_Cedar_S_05": Tree(10, True, 0.7),
    # "Prop_Tree_Cedar_S_06": Tree(6, True, 0.6),
    # "Prop_Tree_Cypress_01": Tree(18, True, 0.7),
    "Prop_Tree_Eng_Oak_01": Tree(17, True, 1.13),
    "Prop_Tree_Eucalip_01": Tree(32, True, 1.51, -0.4),
    "Prop_Tree_Fallen_Pine_01": Tree(11, False, 1.25),
    "Prop_Tree_Jacada_01": Tree(14, True, 0.56),
    "Prop_Tree_Jacada_02": Tree(11, True, 0.71),
    "Prop_Tree_Maple_02": Tree(8, False, 0.41),
    "Prop_Tree_Maple_03": Tree(9, False, 0.35),
    "Prop_Tree_Mquite_01": Tree(5, False, 0.42),  # wrong bounding box
    "Prop_Tree_Oak_01": Tree(23, True, 2.68, -0.1),
    "Prop_Tree_Pine_01": Tree(31, False, 0.82),
    "Prop_Tree_Pine_02": Tree(29, False, 0.80),
    # "Prop_Tree_Stump_01": Tree(2, False, 0.70),
    "Prop_W_R_Cedar_01": Tree(25, False, 1.34),
    "Prop_W_R_Cedar_Dead": Tree(22, False, 1.34),
    "TEST_Tree_Cedar_Trunk_001": Tree(35, False, 1.03),
    "TEST_Tree_Forest_Trunk_01": Tree(104, True, 4.82, -0.5),
    # "TEST_Tree_Forest_Trunk_Base_01": Tree(3, True, 4.25),
}

trees_reduced_probability = {
    "Prop_S_Pine_Dead_01",
    "Prop_Tree_Fallen_Pine_01",
    "Prop_W_R_Cedar_Dead"
}

trees_giant = set()
trees_xlarge = set()
trees_large = set()
trees_medium = set()
trees_small = set()
trees_xsmall = set()
trees_only_flatlands = set()

for treeName in trees.keys():
    tree = trees[treeName]
    if tree.height > 100:
        trees_giant.add(treeName)
    elif tree.height >= 29:
        trees_xlarge.add(treeName)
    elif tree.height >= 20:
        trees_large.add(treeName)
    elif tree.height >= 10:
        trees_medium.add(treeName)
    elif tree.height >= 6:
        trees_small.add(treeName)
    else:
        trees_xsmall.add(treeName)

    if tree.onlyFlatlands:
        trees_only_flatlands.add(treeName)

trees_xlarge_large = trees_xlarge.union(trees_large)

countFlat = 0
countXSmall = 0
countSmall = 0
countMedium = 0
countLarge = 0
countXLarge = 0
countRedwood = 0
countTrees = {}
for tree in trees.keys():
    countTrees[tree] = 0

generatedDir = os.path.join(os.path.dirname(__file__), "generated")
if os.path.exists(generatedDir):
    shutil.rmtree(generatedDir)
os.mkdir(generatedDir)

heightmap = open(os.path.join(os.path.dirname(__file__), 'heights', 'hmap.txt'), 'r')


def getHeights():
    global heightmap

    heightmapEntry = heightmap.readline()
    if not heightmapEntry:
        print("ERROR: cannot get entry in heightmap")
        quit()

    steps = heightmapEntry.split(";")

    if len(steps) < 16:
        print("ERROR: wrong number of values in heightmap entry: " + heightmapEntry)
        quit()

    result = []

    for step in steps:
        minMax = step.split(",")

        if len(minMax) != 2:
            print("ERROR: wrong number of min/max in heightmap entry: " + heightmapEntry)
            quit()

        result.append([float(minMax[0]), float(minMax[1])])

    return result


def floatToStr(val):
    return "{:.8f}".format(val)


def replaceName(content, name):
    return re.sub('(?<=<CMapData>)(\\s*<name>)[^<]+(?=</name>)', "\\g<1>" + name, content)


def repl(matchobj):
    global trees, countFlat, countXSmall, countSmall, countMedium, countLarge, countXLarge, countRedwood

    heights = getHeights()

    origAssetRedwood = (matchobj.group(2) in trees_giant)
    origScaleZ = float(matchobj.group(7))
    origHeight = trees[matchobj.group(2)].height * origScaleZ

    isFlat = (heights[16][1] - heights[16][0] < 3)

    if origAssetRedwood and origHeight >= 100:
        if isFlat and heights[8][0] < random.uniform(120, 140):
            props = trees_giant
        else:
            props = trees_xlarge
    elif origHeight >= 40:
        rand = random.randint(0, 9)
        if isFlat and heights[0][0] < 170 and rand < 2:
            props = trees_medium
        elif isFlat and heights[0][0] < 170 and rand == 2:
            props = trees_small.union(trees_xsmall)
        elif origAssetRedwood:
            props = trees_giant.union(trees_xlarge_large)
        else:
            props = trees_xlarge_large
    elif origHeight >= 28:
        props = trees_medium
        if not isFlat:
            props = props.union(trees_large)
    else:
        props = trees_small.union(trees_xsmall)

    if isFlat:
        countFlat += 1
    else:
        props = props - trees_only_flatlands

    if heights[3][0] > random.uniform(120, 140):
        props = props - {"Prop_Tree_Eucalip_01"}

    if random.random() < 0.5:
        props = props - trees_reduced_probability

    # need to be sorted to get reproducible results in random.choice() for a fixed seed
    choices = sorted(list(props))

    prop = random.choice(choices)

    if prop in trees_giant:
        if origHeight < 60:
            scaleXY = scaleZ = random.uniform(0.3, 0.6)
        else:
            scaleXY = scaleZ = random.uniform(0.7, 1.1)
    elif prop == "TEST_Tree_Forest_Trunk_Base_01":
        scaleXY = scaleZ = random.uniform(0.7, 1.1)
    elif prop == "Prop_Tree_Oak_01":
        scaleXY = scaleZ = random.uniform(1, 1.5)
    elif prop == "Prop_Tree_Eucalip_01":
        scaleXY = scaleZ = random.uniform(1, 1.8)
    elif prop in trees_xlarge_large:
        adaptedOrigScaleZ = origHeight / trees[prop].height
        adaptedOrigScaleZ = max(1.1, min(1.9, adaptedOrigScaleZ))
        scaleZ = random.uniform(adaptedOrigScaleZ - 0.1, adaptedOrigScaleZ + 0.1)
        scaleXY = scaleZ * random.uniform(1, 1.4)
    else:
        scaleZ = random.uniform(1, 2)
        scaleXY = scaleZ * random.uniform(1, 1.2)

    scaleXY = round(scaleXY, 4)
    scaleZ = round(scaleZ, 4)

    tree = trees[prop]
    zCoord = heights[math.ceil(tree.trunkRadius * scaleXY * 2)][0] + tree.offsetZ * scaleZ
    lodDistance = round(tree.getLodDistance(scaleZ))

    countTrees[prop] += 1
    if prop in trees_giant:
        countRedwood += 1
    elif prop in trees_xlarge:
        countXLarge += 1
    elif prop in trees_large:
        countLarge += 1
    elif prop in trees_medium:
        countMedium += 1
    elif prop in trees_small:
        countSmall += 1
    elif prop in trees_xsmall:
        countXSmall += 1
    else:
        print(prop)

    return matchobj.group(1) + prop + matchobj.group(3) + floatToStr(zCoord) + \
           matchobj.group(5) + floatToStr(scaleXY) + matchobj.group(6) + floatToStr(scaleZ) + matchobj.group(8) + \
           floatToStr(lodDistance) + matchobj.group(9)


# create high quality files
for filename in natsorted(os.listdir(os.path.join(os.path.dirname(__file__), "maps"))):
    if not filename.endswith(".ymap.xml") or filename.endswith("_lod.ymap.xml"):
        continue

    f = open(os.path.join(os.path.dirname(__file__), "maps", filename), 'r')
    content = f.read()
    f.close()

    content_new = re.sub('(<Item type="CEntityDef">' +
                         '\\s*<archetypeName>)(TEST_Tree_Forest_Trunk_01)(</archetypeName>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*<position[^>]*\\s+z=")([^"]+)("\\s*/>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*<scaleXY\\s+value=")[^"]+("\\s*/>' +
                         '\\s*<scaleZ\\s+value=")([^"]*)("\\s*/>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*<lodDist value=")[^"]+("\\s*/>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*</Item>)', repl, content, flags=re.M)

    f = open(os.path.join(os.path.dirname(__file__), "generated", filename), 'w')
    f.write(content_new)
    f.close()

for tree in sorted(list(trees.keys())):
    print(tree + ": " + str(countTrees[tree]))

print("")

print("is flat: " + str(countFlat))
print("X-Small: " + str(countXSmall))
print("Small: " + str(countSmall))
print("Medium: " + str(countMedium))
print("Large: " + str(countLarge))
print("X-Large: " + str(countXLarge))
print("Redwood: " + str(countRedwood))

# copy all other files
for filename in os.listdir(os.path.join(os.path.dirname(__file__), "maps")):
    if filename.endswith(".ymap.xml"):
        continue

    shutil.copyfile(os.path.join(os.path.dirname(__file__), "maps", filename), os.path.join(os.path.dirname(__file__), "generated", filename))
