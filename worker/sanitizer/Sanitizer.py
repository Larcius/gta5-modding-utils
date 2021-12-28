from re import Match

import numpy as np
import transforms3d

import os
import re

from common.Util import Util
from common.ymap.Flag import Flag
from common.ymap.Ymap import Ymap
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class Sanitizer:
    # REMOVE_ENTITIES = [
    #    "Prop_Rub_Binbag.*",
    #    "Prop_Rub_BoxPile.*",
    #    "Prop_Rub_Cardpile.*",
    #    "Prop_rub_Flotsam.*",
    #    "Prop_rub_litter.*",
    #    # "Prop_BoxPile.*",
    #    "ng_proc_binbag.*",
    #    "bkr_prop_fakeid_binbag.*",
    #    "hei_prop_heist_binbag.*",
    #    "prop_cs_rub_binbag.*",
    #    "prop_cs_street_binbag.*",
    #    "prop_ld_binbag.*",
    #    "prop_ld_rub_binbag.*",
    #    # "Prop_Shrub_Rake",
    #    # "Prop_Rub_Bike_02",
    #    # "Prop_Rub_cabinet02",
    #    # "Prop_Rub_Scrap_05",
    #    # "Prop_Rub_Tyre_01",
    #    # "Prop_Rub_Stool"
    # ]
    # removeEntitiesPattern = re.compile("(?:" + ")|(?:".join(REMOVE_ENTITIES) + ")", re.IGNORECASE)

    identityQuaternion = [1, -0, -0, -0]

    inputDir: str
    outputDir: str
    ytypItems: dict[str, YtypItem]

    def __init__(self, inputDir: str, outputDir: str):
        self.inputDir = inputDir
        self.outputDir = outputDir

    def run(self):
        print("running sanitizer...")
        self.createOutputDir()
        self.readYtypItems()
        self.processFiles()
        self.copyOthers()
        print("sanitizer DONE")

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)

    def readYtypItems(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "ytyp"))

    def replaceName(self, content: str, name: str) -> str:
        result = re.sub('(?<=<CMapData>)(\\s*<name>)[^<]+(?=</name>)', "\\g<1>" + name, content)
        result = re.sub('(?<=<block>)(' +
                        '(?:\\s*<[^/].*>)*?' +
                        '\\s*<name>)[^<]+(</name>' +
                        '(?:\\s*<[^/].*>)*?' +
                        '\\s*)(?=</block>)', "\\g<1>" + name + "\\g<2>", result)
        return result

    def repl(self, match: Match) -> str:
        archetypeName = match.group(2)

        # if Sanitizer.removeEntitiesPattern.match(archetypeName):
        #    return ""

        flags = int(match.group(3))
        origQuat = [float(match.group(8)), -float(match.group(5)), -float(match.group(6)), -float(match.group(7))]

        rotationQuaternion = np.divide(origQuat, [transforms3d.quaternions.qnorm(origQuat)])

        axangle = transforms3d.quaternions.quat2axangle(rotationQuaternion)
        rotationQuaternion = transforms3d.quaternions.axangle2quat(axangle[0], axangle[1])

        if np.allclose(rotationQuaternion, origQuat, rtol=0, atol=1e-05):
            rotationQuaternion = origQuat
        else:
            print("\t\tfixed rotation:", origQuat, rotationQuaternion)

        if transforms3d.quaternions.nearly_equivalent(rotationQuaternion, Sanitizer.identityQuaternion, rtol=0, atol=1e-05):
            rotationQuaternion = Sanitizer.identityQuaternion
        # do not remove this flag. it is required for dynamic objects as well (even if initially there is no custom orientation)
        # flags &= ~FLAG_ALLOW_FULL_ROTATION
        else:
            # TODO when is it necessary to add this flag? looking at some original rockstar maps only some rotations need this flag
            flags |= Flag.ALLOW_FULL_ROTATION

        return match.group(1) + \
               str(flags) + \
               match.group(4) + \
               'x="' + Util.floatToStr(-rotationQuaternion[1]) + \
               '" y="' + Util.floatToStr(-rotationQuaternion[2]) + \
               '" z="' + Util.floatToStr(-rotationQuaternion[3]) + \
               '" w="' + Util.floatToStr(rotationQuaternion[0]) + '"' + \
               match.group(9) + Util.floatToStr(0) + match.group(10)

    def processFiles(self):
        for filename in os.listdir(self.inputDir):
            if filename.endswith(".ymap.xml"):
                self.processFile(filename)

    def processFile(self, filename: str):
        print("\tprocessing " + filename)

        f = open(os.path.join(self.inputDir, filename), 'r')
        content = f.read()
        f.close()

        content_new = re.sub('(<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '\\s*<flags value=")([^"]+)("\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<rotation )x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"(/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<childLodDist value=")[^"]+("/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*</Item>)', self.repl, content, flags=re.M)

        content_new = self.replaceName(content_new, filename.lower()[:-9])
        content_new = Ymap.calculateAndReplaceLodDistance(content_new, self.ytypItems)
        content_new = Ymap.fixMapExtents(content_new, self.ytypItems)

        f = open(os.path.join(self.outputDir, filename.lower()), 'w')
        f.write(content_new)
        f.close()

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.outputDir, lambda filename: not filename.endswith(".ymap.xml"))
