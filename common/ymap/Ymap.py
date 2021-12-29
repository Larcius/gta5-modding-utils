import math
import re
from datetime import datetime
from re import Match

from common.Util import Util
from common.ymap.Extents import Extents
from common.ymap.PriorityLevel import PriorityLevel
from common.ytyp.YtypItem import YtypItem


class Ymap:
    @staticmethod
    def _replCalculateAndReplaceLodDistance(match: Match, ytypItems: dict[str, YtypItem], onlyIfLodModelExists: bool):
        archetypeName = match.group(2)
        archetypeNameLod = archetypeName + "_LOD"

        # TODO nicht nur onlyIfLodModelExists auswerten, sondern allgemein handhaben anhand von parentIndex
        hasParent = onlyIfLodModelExists

        if onlyIfLodModelExists and archetypeNameLod not in ytypItems:
            return match.group(0)

        scale = [float(match.group(3)), float(match.group(3)), float(match.group(4))]

        if archetypeName in ytypItems:
            lodDistance = ytypItems[archetypeName].getLodDistance(scale, hasParent)
            lodDistance = math.ceil(lodDistance)
        elif archetypeNameLod in ytypItems:
            lodDistance = ytypItems[archetypeNameLod].getLodDistance(scale, hasParent)
            lodDistance = math.ceil(lodDistance)
        else:
            print("WARNING: could not find archetype " + archetypeName + " in any of the provided ytyp files. Leaving lodDistance for those unchanged.")
            lodDistance = float(match.group(5))

        priorityLevel = PriorityLevel.getLevel(lodDistance, hasParent)
        if priorityLevel != PriorityLevel.REQUIRED:
            # for optional entities use -1 to indicate that the lod distance should be automatically determined
            # (as seen in original Rockstar ymap files)
            lodDistance = -1

        return match.group(1) + Util.floatToStr(lodDistance) + match.group(6) + priorityLevel + match.group(7)

    @staticmethod
    def calculateAndReplaceLodDistanceForEntitiesWithLod(contentNoLod: str, ytypItems: dict[str, YtypItem]) -> str:
        pattern = re.compile('(\\s*<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<scaleXY value="([^"]+)"\\s*/>' +
                             '\\s*<scaleZ value="([^"]+)"\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<lodDist value=")([^"]+)("\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<priorityLevel>)[^<]*(</priorityLevel>'
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*</Item>)', flags=re.M)

        return pattern.sub(lambda match: Ymap._replCalculateAndReplaceLodDistance(match, ytypItems, True), contentNoLod)

    @staticmethod
    def calculateAndReplaceLodDistance(contentNoLod: str, ytypItems: dict[str, YtypItem]) -> str:
        pattern = re.compile('(\\s*<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<scaleXY value="([^"]+)"\\s*/>' +
                             '\\s*<scaleZ value="([^"]+)"\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<lodDist value=")([^"]+)("\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<priorityLevel>)[^<]*(</priorityLevel>'
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*</Item>)', flags=re.M)

        return pattern.sub(lambda match: Ymap._replCalculateAndReplaceLodDistance(match, ytypItems, False), contentNoLod)

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

    # adapt extents and set current datetime
    @staticmethod
    def fixMapExtents(content: str, ytypItems: dict[str, YtypItem]) -> str:
            extents = Extents.calculateExtents(content, ytypItems)

            if extents.isValid():
                result = extents.replaceExtents(content)
            else:
                result = content

            nowLocalIso = datetime.now().astimezone().replace(microsecond=0).isoformat()
            return Ymap.replaceDatetime(result, nowLocalIso)
