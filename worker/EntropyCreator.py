import numpy as np
import transforms3d
import os
import math
import random
import re
from re import Match

from natsort import natsorted

from common.Util import Util
from common.ymap.Ymap import Ymap
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class EntropyCreator:
    CANDIDATES_SCALE = tuple({
        "prop_palm_med",
        "prop_palm_huge",
        "test_tree_forest",
    })
    CANDIDATES_ROTATION = tuple({
        "prop_cactus",
        "prop_joshua_tree",
        "prop_joshua_tree",
        "prop_palm_fan",
        "prop_palm_huge",
        "prop_palm_med",
        "prop_palm_sm",
        "prop_s_pine_dead",
        "prop_tree_birch",
        "prop_tree_cedar",
        "prop_tree_cypress",
        "prop_tree_eng_oak",
        "prop_tree_eucalip",
        "prop_tree_fallen_pine",
        "prop_tree_jacada",
        "prop_tree_lficus",
        "prop_tree_maple",
        "prop_tree_oak",
        "prop_tree_olive",
        "prop_tree_pine",
        "prop_w_r_cedar",
        "test_tree",
    })


    inputDir: str
    outputDir: str
    ytypItems: dict[str, YtypItem]
    limitTilt: bool
    adaptRotationIfIdentity: bool
    limitScale: bool

    def __init__(self, inputDir: str, outputDir: str, limitTilt: bool, adaptRotationIfIdentity: bool, limitScale: bool):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.limitTilt = limitTilt
        self.adaptRotationIfIdentity = adaptRotationIfIdentity
        self.limitScale = limitScale

        # using a specific seed to be able to get reproducible results
        random.seed(a=0)

    def run(self):
        print("running entropy creator...")
        self.createOutputDir()
        self.readYtypItems()
        self.processFiles()
        self.copyOthers()
        print("entropy creator DONE")

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)

    def readYtypItems(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "..", "resources", "ytyp"))

    def isScaleCandidate(self, entity: str) -> bool:
        return entity in self.ytypItems and entity.lower().startswith(EntropyCreator.CANDIDATES_SCALE)

    def isRotationCandidate(self, entity: str) -> bool:
        return entity in self.ytypItems and entity.lower().startswith(EntropyCreator.CANDIDATES_ROTATION)

    def repl(self, match: Match) -> str:
        entity = match.group(2).lower()
        origScaleXY = float(match.group(8))
        origScaleZ = float(match.group(10))

        maxScaleZ = random.uniform(0.85, 1.1)

        isRotationCandidate = self.isRotationCandidate(entity)
        isScaleCandidate = self.isScaleCandidate(entity)

        if self.limitScale and isScaleCandidate and origScaleZ > maxScaleZ:
            ratio = maxScaleZ / origScaleZ
            scaleZ = maxScaleZ
            scaleXY = origScaleXY * ratio
        else:
            scaleXY = origScaleXY
            scaleZ = origScaleZ

        if self.limitTilt and isRotationCandidate:
            treeHeight = self.ytypItems[entity].boundingBox.getSizes()[2] * scaleZ
            maxAngleNoZ = Util.calculateMaxTilt(treeHeight)
        else:
            maxAngleNoZ = math.pi

        origQuat = [float(match.group(6)), -float(match.group(3)), -float(match.group(4)), -float(match.group(5))]  # order is w, -x, -y, -z

        origRotZ, origRotY, origRotX = transforms3d.euler.quat2euler(origQuat, axes='rzyx')

        axisNoZ, angleNoZ = transforms3d.euler.euler2axangle(0, origRotY, origRotX, axes='rzyx')

        if self.limitTilt and abs(angleNoZ) > maxAngleNoZ:
            newAngleNoZ = math.copysign(maxAngleNoZ, angleNoZ)
        else:
            newAngleNoZ = angleNoZ

        rotZ = origRotZ
        if self.adaptRotationIfIdentity and isRotationCandidate:
            if origRotZ == 0:
                rotZ = random.uniform(-math.pi, math.pi)
            if angleNoZ == 0:
                defaultMaxTilt = Util.calculateMaxTilt(999)
                newAngleNoZ = random.uniform(-defaultMaxTilt, defaultMaxTilt)

        if self.limitTilt or self.adaptRotationIfIdentity:
            unused, rotY, rotX = transforms3d.euler.axangle2euler(axisNoZ, newAngleNoZ, axes='rzyx')

            rotationQuaternion = transforms3d.euler.euler2quat(rotZ, rotY, rotX, axes='rzyx')
        else:
            rotationQuaternion = origQuat

        if np.allclose(rotationQuaternion, origQuat, rtol=0, atol=1e-05) and np.allclose([scaleXY, scaleZ], [origScaleXY, origScaleZ], rtol=0, atol=1e-05):
            return match.group(0)

        if self.limitTilt and abs(angleNoZ) > maxAngleNoZ:
            print("\t\tlimiting original tilt (angle ignoring rotation around z axis) of " + Util.angleToStr(angleNoZ) + "° to " +
                  Util.angleToStr(newAngleNoZ) + "°")

        if not np.allclose(rotationQuaternion, origQuat, rtol=0, atol=1e-05):
            print("\t\tchanging rotation from tilt " + Util.angleToStr(angleNoZ) + "° to " + Util.angleToStr(newAngleNoZ) + "° and yaw " +
                  Util.angleToStr(origRotZ) + "° to " + Util.angleToStr(rotZ) + "°")
        else:
            rotationQuaternion = origQuat

        if not np.allclose([scaleXY, scaleZ], [origScaleXY, origScaleZ], rtol=0, atol=1e-05):
            print("\t\tchanging scale from " + Util.floatToStr(origScaleXY) + ", " + Util.floatToStr(origScaleZ) + " to " +
                  Util.floatToStr(scaleXY) + ", " + Util.floatToStr(scaleZ))
        else:
            scaleXY = origScaleXY
            scaleZ = origScaleZ

        return match.group(1) + \
               'x="' + Util.floatToStr(-rotationQuaternion[1]) + '" y="' + Util.floatToStr(-rotationQuaternion[2]) + \
               '" z="' + Util.floatToStr(-rotationQuaternion[3]) + '" w="' + Util.floatToStr(rotationQuaternion[0]) + '"' + \
               match.group(7) + Util.floatToStr(scaleXY) + match.group(9) + Util.floatToStr(scaleZ) + match.group(11)

    def processFiles(self):
        for filename in natsorted(os.listdir(self.inputDir)):
            if filename.endswith(".ymap.xml"):
                self.processFile(filename)

    def processFile(self, filename: str):
        print("\tprocessing " + filename)

        f = open(os.path.join(self.inputDir, filename), 'r')
        content = f.read()
        f.close()

        content_new = re.sub('(<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<rotation )x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"(/>' +
                             '\\s*<scaleXY value=")([^"]+)("/>' +
                             '\\s*<scaleZ value=")([^"]+)("/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*</Item>)', self.repl, content, flags=re.M)

        content_new = Ymap.fixMapExtents(content_new, self.ytypItems)

        f = open(os.path.join(self.outputDir, filename.lower()), 'w')
        f.write(content_new)
        f.close()

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.outputDir, lambda filename: not filename.endswith(".ymap.xml"))
