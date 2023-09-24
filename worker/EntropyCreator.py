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
        "prop_desert_iron",
        "prop_rio_del",
        "prop_rus_olive",
        "prop_fan_palm",
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
        "prop_tree_mquite",
        "prop_tree_oak",
        "prop_tree_olive",
        "prop_tree_pine",
        "prop_w_r_cedar",
        "test_tree",
        "prop_tree_stump",
        "prop_veg_crop_orange",
        "prop_bush_lrg_04"
    })
    CANDIDATES_ROTATION = tuple({
        "prop_desert_iron",
        "prop_rio_del",
        "prop_rus_olive",
        "prop_cactus",
        "prop_joshua_tree",
        "prop_fan_palm",
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
        "prop_tree_mquite",
        "prop_tree_oak",
        "prop_tree_olive",
        "prop_tree_pine",
        "prop_w_r_cedar",
        "test_tree",
        "prop_tree_stump",
        "prop_veg_crop_orange",
        "prop_bush_lrg_04"
    })


    inputDir: str
    outputDir: str
    ytypItems: dict[str, YtypItem]
    limitTilt: bool
    adaptRotationIfIdentity: bool
    limitScale: bool
    adaptScaleIfIdentity: bool

    def __init__(self, inputDir: str, outputDir: str, limitTilt: bool, adaptRotationIfIdentity: bool, limitScale: bool, adaptScaleIfIdentity: bool):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.limitTilt = limitTilt
        self.adaptRotationIfIdentity = adaptRotationIfIdentity
        self.limitScale = limitScale
        self.adaptScaleIfIdentity = adaptScaleIfIdentity

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
        return entity in self.ytypItems and entity.startswith(EntropyCreator.CANDIDATES_SCALE)

    def isRotationCandidate(self, entity: str) -> bool:
        return entity in self.ytypItems and entity.startswith(EntropyCreator.CANDIDATES_ROTATION)

    def repl(self, match: Match) -> str:
        entity = match.group(2).lower()

        origScale = [float(match.group(8)), float(match.group(10))]
        scale = self.adaptScale(entity, origScale)

        origQuat = [float(match.group(6)), -float(match.group(3)), -float(match.group(4)), -float(match.group(5))]  # order is w, -x, -y, -z
        rotationQuaternion = self.adaptRotation(entity, origQuat, scale[1])

        if scale == origScale and rotationQuaternion == origQuat:
            return match.group(0)

        return match.group(1) + \
               'x="' + Util.floatToStr(-rotationQuaternion[1]) + '" y="' + Util.floatToStr(-rotationQuaternion[2]) + \
               '" z="' + Util.floatToStr(-rotationQuaternion[3]) + '" w="' + Util.floatToStr(rotationQuaternion[0]) + '"' + \
               match.group(7) + Util.floatToStr(scale[0]) + match.group(9) + Util.floatToStr(scale[1]) + match.group(11)

    def adaptScale(self, entity: str, origScale: list[float]) -> list[float]:
        if not self.isScaleCandidate(entity):
            return origScale

        maxScaleZ = 1.1
        if self.adaptScaleIfIdentity and origScale == [1, 1]:
            scaleXY = scaleZ = random.uniform(1 / maxScaleZ, maxScaleZ)
        elif self.limitScale and origScale[1] > maxScaleZ:
            ratio = maxScaleZ / origScale[1]
            scaleZ = maxScaleZ
            scaleXY = origScale[0] * ratio
        else:
            return origScale

        print("\t\tchanging scale from " + Util.floatToStr(origScale[0]) + ", " + Util.floatToStr(origScale[1]) + " to " +
              Util.floatToStr(scaleXY) + ", " + Util.floatToStr(scaleZ))

        return [scaleXY, scaleZ]

    def adaptRotation(self, entity: str, origQuat: list[float], scaleZ: float) -> list[float]:
        if not self.isRotationCandidate(entity):
            return origQuat

        if self.adaptRotationIfIdentity and origQuat == [1, 0, 0, 0]:
            maxTilt = self.calculateMaxTilt(entity, scaleZ)
            rotZBefore = random.uniform(-math.pi, math.pi)
            tilt = random.uniform(0, maxTilt)
            rotZAfter = random.uniform(-math.pi, math.pi)

            rotationQuaternion = transforms3d.euler.euler2quat(rotZBefore, tilt, rotZAfter, axes='rzyz')
        elif self.limitTilt:
            maxTilt = self.calculateMaxTilt(entity, scaleZ)
            rotZBefore, origTilt, rotZAfter = transforms3d.euler.quat2euler(origQuat, axes='rzyz')

            if abs(origTilt) < maxTilt:
                return origQuat

            tilt = math.copysign(maxTilt, origTilt)

            print("\t\tlimiting original tilt of " +
                  Util.angleToStr(abs(origTilt)) + " to " + Util.angleToStr(abs(tilt)))

            rotationQuaternion = transforms3d.euler.euler2quat(rotZBefore, tilt, rotZAfter, axes='rzyz')
        else:
            return origQuat

        origRotY, origRotX, origRotZ = transforms3d.euler.quat2euler(origQuat, axes='syxz')
        rotY, rotX, rotZ = transforms3d.euler.quat2euler(rotationQuaternion, axes='syxz')
        print("\t\tchanging rotation from " +
              Util.angleToStr(-origRotX) + ", " + Util.angleToStr(-origRotY) + ", " + Util.angleToStr(-origRotZ) + " to " +
              Util.angleToStr(-rotX) + ", " + Util.angleToStr(-rotY) + ", " + Util.angleToStr(-rotZ))

        return rotationQuaternion.tolist()

    def calculateMaxTilt(self, entity: str, scaleZ: float) -> float:
        treeHeight = self.ytypItems[entity].boundingBox.getSizes()[2] * scaleZ
        return Util.calculateMaxTilt(treeHeight)

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
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<rotation )x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"(/>' +
                             '\\s*<scaleXY value=")([^"]+)("/>' +
                             '\\s*<scaleZ value=")([^"]+)("/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*</Item>)', self.repl, content, flags=re.M)

        content_new = Ymap.fixMapExtents(content_new, self.ytypItems)

        f = open(os.path.join(self.outputDir, filename.lower()), 'w')
        f.write(content_new)
        f.close()

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.outputDir, lambda filename: not filename.endswith(".ymap.xml"))
