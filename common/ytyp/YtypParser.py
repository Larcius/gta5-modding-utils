import os
import re

from common.Box import Box
from common.Sphere import Sphere
from common.ytyp.YtypItem import YtypItem


class YtypParser:
    @staticmethod
    def getExpressionYtypItem() -> str:
        return '\\s*<Item type="CBaseArchetypeDef">' + \
               '\\s*<lodDist value="([^"]+)"/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<bbMin x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' + \
               '\\s*<bbMax x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' + \
               '\\s*<bsCentre x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' + \
               '\\s*<bsRadius value="([^"]+)"/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<name>([^<]+)</name>'

    @staticmethod
    def readYtypDirectory(path: str) -> dict[str, YtypItem]:
        items = {}

        if not os.path.exists(path):
            return items

        for filename in os.listdir(path):
            if not filename.endswith(".ytyp.xml"):
                continue

            items |= YtypParser.readYtypFile(os.path.join(path, filename))

        return items

    @staticmethod
    def readYtypFile(ytypFile: str) -> dict[str, YtypItem]:
        f = open(ytypFile, 'r')
        content = f.read()
        f.close()

        return YtypParser.readYtypContent(content)

    @staticmethod
    def readYtypContent(ytypContent: str) -> dict[str, YtypItem]:
        parent = YtypParser.getYtypName(ytypContent)

        items = {}
        for match in re.finditer(YtypParser.getExpressionYtypItem(), ytypContent):
            lodDist = float(match.group(1))
            bbMin = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            bbMax = [float(match.group(5)), float(match.group(6)), float(match.group(7))]
            bsCenter = [float(match.group(8)), float(match.group(9)), float(match.group(10))]
            bsRadius = float(match.group(11))
            name = match.group(12).lower()
            items[name] = YtypItem(lodDist, Box(bbMin, bbMax), Sphere(bsCenter, bsRadius), parent)

        return items

    @staticmethod
    def getYtypName(ytypContent: str) -> str:
        match = re.search('\\s*<name>([^<]+)</name>' +
        '\\s*(?:<dependencies/>|<dependencies>[\\S\\s]*</dependencies>)' +
        '\\s*(?:<compositeEntityTypes/>|<compositeEntityTypes>[\\S\\s]*</compositeEntityTypes>)' +
        '\\s*</CMapTypes>', ytypContent, re.M)

        return match.group(1)
