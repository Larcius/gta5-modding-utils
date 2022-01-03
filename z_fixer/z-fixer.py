# TODO refactoring needed

import os
import re
import shutil
import math
from dataclasses import dataclass
from natsort import natsorted


@dataclass
class Tree:
    trunkRadius: float
    offsetZ: float = 0


# True means extract x, y coordinates
# False means fix z coordinates
ENABLE_MODE_EXTRACT = False

# if True then z coordinate will not be changed if calculated value is greater than original value
DISABLE_INCREASE_OF_Z = True

# trees can be found at x64i.rpf\levels\gta5\props\vegetation\v_trees.rpf\
# and update\x64\dlcpacks\patchday2ng\dlc.rpf\x64\levels\gta5\props\vegetation\v_trees.rpf

trees = {
    # trees
    "Prop_Rio_Del_01": Tree(1, 0.09),
    "Prop_S_Pine_Dead_01": Tree(0.55),
    "Prop_Tree_Birch_01": Tree(0.95, -0.01),
    "Prop_Tree_Birch_02": Tree(0.63),
    "Prop_Tree_Birch_03": Tree(0.24, -0.03),
    "Prop_Tree_Birch_03b": Tree(0.18, -0.04),
    "Prop_Tree_Birch_04": Tree(0.57, -0.03),
    "Prop_Tree_Birch_05": Tree(0.22, -0.06),
    "Prop_Tree_Cedar_02": Tree(0.89, 0.08),
    "Prop_Tree_Cedar_03": Tree(0.65, 0.08),
    "Prop_Tree_Cedar_04": Tree(1.1, 0.09),
    "Prop_Tree_Cedar_S_01": Tree(0.52, 0.03),
    "Prop_Tree_Cedar_S_02": Tree(0.13, 0.01),
    "Prop_Tree_Cedar_S_04": Tree(1),
    "Prop_Tree_Cedar_S_05": Tree(0.7),
    "Prop_Tree_Cedar_S_06": Tree(0.6),
    "Prop_Tree_Cypress_01": Tree(0.7),
    "Prop_Tree_Eng_Oak_01": Tree(1.13),
    "Prop_Tree_Eucalip_01": Tree(1.51, -0.17),
    "Prop_Tree_Fallen_Pine_01": Tree(1.25, 0.54),
    "Prop_Tree_Jacada_01": Tree(0.56, -0.06),
    "Prop_Tree_Jacada_02": Tree(0.71, 0.1),
    "Prop_Tree_Maple_02": Tree(0.41, 0.05),
    "Prop_Tree_Maple_03": Tree(0.35, 0.05),
    "Prop_Tree_Mquite_01": Tree(0.45, -0.03),
    "Prop_Tree_Oak_01": Tree(2.68, -0.07),
    "Prop_Tree_Pine_01": Tree(0.82, 0.44),
    "Prop_Tree_Pine_02": Tree(0.80, -0.04),
    "Prop_Tree_Stump_01": Tree(0.70, 0.02),
    "Prop_W_R_Cedar_01": Tree(1.34, 0.09),
    "Prop_W_R_Cedar_Dead": Tree(1.34, 0.09),
    "TEST_Tree_Cedar_Trunk_001": Tree(1.03, 0.09),
    "TEST_Tree_Forest_Trunk_01": Tree(4.82, -0.03),
    "TEST_Tree_Forest_Trunk_Base_01": Tree(4.25),
    # bushes
    "Prop_Bush_Lrg_02": Tree(3, 0.188),
    "Prop_Bush_Lrg_02b": Tree(1.1, -0.01),
    "Prop_Bush_Lrg_04b": Tree(2.6, 2.2),
    "Prop_Bush_Lrg_04c": Tree(2.6, 2.2),
    "Prop_Bush_Lrg_04d": Tree(2.6, 2.2),
    # palms
    "Prop_Palm_Sm_01d": Tree(0.7, -0.05),
    "Prop_Palm_Sm_01e": Tree(1.1, -0.05),
    "Prop_Palm_Sm_01f": Tree(0.72),
    "Prop_Palm_Med_01a": Tree(0.89),
    "Prop_Palm_Med_01b": Tree(1, -0.05),
    "Prop_Palm_Med_01c": Tree(1.2, -0.1),
    "Prop_Fan_Palm_01a": Tree(1.2, 0.23),
    "Prop_Palm_Fan_02_a": Tree(0.6, -0.05),
    "Prop_Palm_Fan_02_b": Tree(1.35, -0.1),
    "Prop_Palm_Sm_01a": Tree(0.65, -0.05),
    "Prop_Palm_Fan_04_a": Tree(0.65, -0.1),
    "Prop_Palm_Fan_04_b": Tree(0.85, -0.12),
    "Prop_Palm_Fan_03_a": Tree(1.7, -0.03),
    "Prop_Palm_Fan_03_b": Tree(0.9, -0.03),
    "Prop_Palm_Fan_03_c": Tree(1),
    "Prop_Palm_Fan_03_c_graff": Tree(1),
    "Prop_Palm_Fan_04_c": Tree(1.25, -0.22),
    "Prop_Palm_Med_01d": Tree(1.15),
    "Prop_Palm_Fan_03_d": Tree(0.8),
    "Prop_Palm_Fan_03_d_Graff": Tree(0.8),
    "Prop_Palm_Fan_04_d": Tree(1.5, -0.15),
    "Prop_Palm_Huge_01a": Tree(0.95, 0.37),
    "Prop_Palm_Huge_01b": Tree(0.95, -0.08),
}

generatedDir = os.path.join(os.path.dirname(__file__), "generated")
if os.path.exists(generatedDir):
    shutil.rmtree(generatedDir)
os.mkdir(generatedDir)

if ENABLE_MODE_EXTRACT:
    outCoords = open(os.path.join(generatedDir, "coords.txt"), 'w')
else:
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


def repl(matchobj):
    global trees

    prop = matchobj.group(2)

    if prop not in trees:
        return matchobj.group(0)

    if ENABLE_MODE_EXTRACT:
        outCoords.write(matchobj.group(3) + ", " + matchobj.group(4) + "\n")
        return matchobj.group(0)

    heights = getHeights()

    zCoord = float(matchobj.group(5))
    scaleXY = float(matchobj.group(7))
    scaleZ = float(matchobj.group(8))

    tree = trees[prop]
    calcZCoord = heights[math.ceil(tree.trunkRadius * scaleXY * 2)][0] + tree.offsetZ * scaleZ

    if DISABLE_INCREASE_OF_Z and calcZCoord > zCoord:
        calcZCoord = zCoord

    return matchobj.group(1) + floatToStr(calcZCoord) + matchobj.group(6)


# create high quality files
for filename in natsorted(os.listdir(os.path.join(os.path.dirname(__file__), "maps"))):
    if not filename.endswith(".ymap.xml") or filename.endswith("_lod.ymap.xml"):
        continue

    f = open(os.path.join(os.path.dirname(__file__), "maps", filename), 'r')
    content = f.read()
    f.close()

    content_new = re.sub('(<Item type="CEntityDef">' +
                         '\\s*<archetypeName>([^<]+)</archetypeName>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*<position x="([^>]+)" y="([^>]+)" z=")([^"]+)("\\s*/>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*<scaleXY\s+value="([^"]+)"\\s*/>' +
                         '\\s*<scaleZ\s+value="([^"]+)"\\s*/>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*</Item>)', repl, content, flags=re.M)

    if not ENABLE_MODE_EXTRACT:
        f = open(os.path.join(os.path.dirname(__file__), "generated", filename), 'w')
        f.write(content_new)
        f.close()

if not ENABLE_MODE_EXTRACT:
    heightmap.close()

    # copy all other files
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), "maps")):
        if filename.endswith(".ymap.xml"):
            continue

        shutil.copyfile(os.path.join(os.path.dirname(__file__), "maps", filename), os.path.join(os.path.dirname(__file__), "generated", filename))
