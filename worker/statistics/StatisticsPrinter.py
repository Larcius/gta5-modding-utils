import os
import re

from natsort import natsorted

from common.ymap.LodLevel import LodLevel
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class StatisticsPrinter:
    countProps: dict[str, dict[str, int]]
    inputDir: str
    ytypItems: dict[str, YtypItem]

    def __init__(self, inputDir: str):
        self.inputDir = inputDir

    def run(self):
        self.readYtypItems()
        self.countProps = {}
        self.processFiles()

    def readYtypItems(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "ytyp"))

    def processFiles(self):
        for filename in os.listdir(self.inputDir):
            if not filename.endswith(".ymap.xml") or filename.endswith("_lod.ymap.xml"):
                continue

            f = open(os.path.join(self.inputDir, filename), 'r')
            content = f.read()

            expression = '<Item type="CEntityDef">' + \
                         '\\s*<archetypeName>([^<]+)</archetypeName>' + \
                         '(?:\\s*<[^/].*>)*?' + \
                         '\\s*<lodLevel>(?:' + LodLevel.HD + "|" + LodLevel.ORPHAN_HD + ')</lodLevel>' + \
                         '(?:\\s*<[^/].*>)*?' + \
                         '\\s*</Item>'

            for match in re.finditer(expression, content):
                archetypeName = match.group(1)

                if archetypeName in self.ytypItems:
                    ytypName = self.ytypItems[archetypeName].parent
                else:
                    ytypName = "others"

                # if not tree.startswith("Prop_S_Pine_") and not tree.startswith("Prop_Tree_") and not tree.startswith("Prop_W_R_Cedar_") and not tree.startswith("TEST_Tree_"):
                #	continue

                if ytypName not in self.countProps:
                    self.countProps[ytypName] = {}

                if archetypeName not in self.countProps[ytypName]:
                    self.countProps[ytypName][archetypeName] = 0

                self.countProps[ytypName][archetypeName] += 1

        totalCount = 0
        ytypCounts = {}
        for ytyp in natsorted(list(self.countProps.keys())):
            ytypCounts[ytyp] = 0
            print(ytyp + ":")
            for prop in natsorted(list(self.countProps[ytyp])):
                num = self.countProps[ytyp][prop]
                ytypCounts[ytyp] += num
                print("\t" + prop + ":\t\t" + str(num))
            totalCount += ytypCounts[ytyp]
            print("\t----------------------------------------------")
            print("\t" + ytyp + " total:\t\t" + str(ytypCounts[ytyp]) + "\n")

        print("\nsummary:")
        for ytyp in natsorted(list(ytypCounts.keys())):
            print(ytyp + ":\t\t" + str(ytypCounts[ytyp]))
        print("----------------------------------------------")
        print("total:\t\t" + str(totalCount))
