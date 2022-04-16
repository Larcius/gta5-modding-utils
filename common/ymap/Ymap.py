import math
import re
from re import Match
from typing import Optional

from common.Util import Util
from common.ymap.Extents import Extents
from common.ymap.PriorityLevel import PriorityLevel
from common.ytyp.YtypItem import YtypItem


class Ymap:
    @staticmethod
    def _replCalculateAndReplaceLodDistance(match: Match, ytypItems: dict[str, YtypItem], archetypes: Optional[list[str]], forceHasParent: bool):
        archetypeName = match.group(2).lower()

        if archetypes is not None and archetypeName not in archetypes:
            return match.group(0)

        hasParent = True if forceHasParent else int(match.group(5)) >= 0

        scale = [float(match.group(3)), float(match.group(3)), float(match.group(4))]

        if archetypeName in ytypItems:
            lodDistance = math.ceil(ytypItems[archetypeName].getLodDistance(scale, hasParent))
        else:
            print("WARNING: could not find archetype " + archetypeName + " in any of the provided ytyp files. Leaving lodDistance for those unchanged.")
            lodDistance = float(match.group(6))

        priorityLevel = PriorityLevel.getLevel(lodDistance, hasParent)
        if priorityLevel != PriorityLevel.REQUIRED or lodDistance < 100:
            # for optional entities use -1 to indicate that the lod distance should be automatically determined
            # (as seen in original Rockstar ymap files)
            lodDistance = -1

        return match.group(1) + Util.floatToStr(lodDistance) + match.group(7) + priorityLevel + match.group(8)

    @staticmethod
    def calculateAndReplaceLodDistance(contentNoLod: str, ytypItems: dict[str, YtypItem], archetypes=None, forceHasParent=False) -> str:
        pattern = re.compile('(\\s*<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<scaleXY value="([^"]+)"\\s*/>' +
                             '\\s*<scaleZ value="([^"]+)"\\s*/>' +
                             '\\s*<parentIndex value="([^"]+)"/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<lodDist value=")([^"]+)("\\s*/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<priorityLevel>)[^<]*(</priorityLevel>'
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*</Item>)', flags=re.M)

        return pattern.sub(lambda match: Ymap._replCalculateAndReplaceLodDistance(match, ytypItems, archetypes, forceHasParent), contentNoLod)

    @staticmethod
    def replaceDatetime(content: str, nowIso: str) -> str:
        return re.sub(
            '(?<=<block>)(' +
            '(?:\\s*<[^/].*>)*?' +
            '\\s*<exportedBy>)[^<]+(</exportedBy>'
            '(?:\\s*<[^/].*>)*?' +
            '\\s*<time>)[^<]+(</time>' +
            '(?:\\s*<[^/].*>)*?' +
            '\\s*)(?=</block>)',
            "\\g<1>Larcius\\g<2>" + nowIso + "\\g<3>", content
        )

    @staticmethod
    def replaceName(content: str, name: str) -> str:
        result = re.sub('(?<=<CMapData>)(\\s*<name>)[^<]+(?=</name>)', "\\g<1>" + name, content)
        result = re.sub('(?<=<block>)(' +
                        '(?:\\s*<[^/].*>)*?' +
                        '\\s*<name>)[^<]+(</name>' +
                        '(?:\\s*<[^/].*>)*?' +
                        '\\s*)(?=</block>)', "\\g<1>" + name + "\\g<2>", result)
        return result

    @staticmethod
    def replaceParent(content: str, parent: Optional[str]) -> str:
        if parent is None or parent == "":
            newParent = "<parent/>"
        else:
            newParent = "<parent>" + parent + "</parent>"
        return re.sub('<parent\\s*(?:/>|>[^<]*</parent>)', newParent, content, flags=re.M)

    # adapt extents and set current datetime
    @staticmethod
    def fixMapExtents(content: str, ytypItems: dict[str, YtypItem]) -> str:
            extents = Extents.calculateExtents(content, ytypItems)

            if extents.isValid():
                result = extents.replaceExtents(content)
            else:
                result = content

            return Ymap.replaceDatetime(result, Util.getNowInIsoFormat())
