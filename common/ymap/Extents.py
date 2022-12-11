import numpy as np
import transforms3d

import re

from common.Box import Box
from common.Util import Util
from common.ytyp.YtypItem import YtypItem


class Extents:
    CARGEN_LOD_DISTANCE = 250

    @staticmethod
    def createReversedInfinityExtents() -> "Extents":
        return Extents(Box.createReversedInfinityBox(), Box.createReversedInfinityBox())

    @staticmethod
    def getExpressionForCalculateExtents() -> str:
        return '<Item type="CEntityDef">' + \
               '\\s*<archetypeName>([^<]+)</archetypeName>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' + \
               '\\s*<rotation x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"/>' + \
               '\\s*<scaleXY value="([^"]+)"/>' + \
               '\\s*<scaleZ value="([^"]+)"/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<lodDist value="([^"]+)"/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*</Item>'

    @staticmethod
    def getExpressionForCalculateExtentsCarGen() -> str:
        return '<Item>' + \
               '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<perpendicularLength value="([^"]+)"/>' + \
               '\\s*<carModel>([^<]+)</carModel>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*</Item>'

    @staticmethod
    def calculateExtents(ymapContent: str, ytypItems: dict[str, YtypItem]) -> "Extents":
        extents = Extents.createReversedInfinityExtents()

        for match in re.finditer(Extents.getExpressionForCalculateExtents(), ymapContent):
            archetypeName = match.group(1).lower()

            if archetypeName not in ytypItems:
                print("WARNING: could not find archetype " + archetypeName + ". Proceeding without it but this might yield wrong extents")
                continue

            position = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
            rotationQuat = [float(match.group(8)), -float(match.group(5)), -float(match.group(6)), -float(match.group(7))]
            scale = [float(match.group(9)), float(match.group(9)), float(match.group(10))]
            lodDistance = float(match.group(11))
            if lodDistance < 0:
                lodDistance = ytypItems[archetypeName].lodDist
            bbox = ytypItems[archetypeName].boundingBox

            extents.adaptExtents(position, rotationQuat, scale, lodDistance, bbox)

        for match in re.finditer(Extents.getExpressionForCalculateExtentsCarGen(), ymapContent):
            perpendicularLength = float(match.group(4))
            carModel = match.group(5).lower()

            print("INFO: found carGenerator for car model " + carModel + ". Using " + str(Extents.CARGEN_LOD_DISTANCE) + " as lodDistance.")

            position = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            bbox = Box.createUnitBox().getScaled([perpendicularLength] * 3)

            extents.adaptExtents(position, [0, 0, 0, 1], [1, 1, 1], Extents.CARGEN_LOD_DISTANCE, bbox)

        return extents

    entities: Box
    streaming: Box

    def __init__(self, entities: Box, streaming: Box):
        self.entities = entities
        self.streaming = streaming

    def replaceExtents(self, contentYmap: str) -> str:
        return re.sub(
            '(<streamingExtentsMin x=")[^"]+(" y=")[^"]+(" z=")[^"]+("/>' +
            '\\s*<streamingExtentsMax x=")[^"]+(" y=")[^"]+(" z=")[^"]+("/>' +
            '\\s*<entitiesExtentsMin x=")[^"]+(" y=")[^"]+(" z=")[^"]+("/>' +
            '\\s*<entitiesExtentsMax x=")[^"]+(" y=")[^"]+(" z=")[^"]+("/>)',
            "\\g<1>" + Util.floatToStr(self.streaming.min[0]) + "\\g<2>" + Util.floatToStr(self.streaming.min[1]) + "\\g<3>" +
            Util.floatToStr(self.streaming.min[2]) +
            "\\g<4>" + Util.floatToStr(self.streaming.max[0]) + "\\g<5>" + Util.floatToStr(self.streaming.max[1]) + "\\g<6>" +
            Util.floatToStr(self.streaming.max[2]) +
            "\\g<7>" + Util.floatToStr(self.entities.min[0]) + "\\g<8>" + Util.floatToStr(self.entities.min[1]) + "\\g<9>" +
            Util.floatToStr(self.entities.min[2]) +
            "\\g<10>" + Util.floatToStr(self.entities.max[0]) + "\\g<11>" + Util.floatToStr(self.entities.max[1]) + "\\g<12>" +
            Util.floatToStr(self.entities.max[2]) + "\\g<13>", contentYmap
        )

    def adaptExtents(self, position: list[float], rotationQuaternion: list[float], scale: list[float], lodDistance: float, bbox: Box):
        scaledBbox = bbox.getScaled(scale)
        scaledLodBbox = scaledBbox.getExtended([lodDistance] * 3)

        scaledBboxList = [scaledBbox.min, scaledBbox.max]
        scaledLodBboxList = [scaledLodBbox.min, scaledLodBbox.max]
        for i in range(8):
            point = [scaledBboxList[i % 2][0], scaledBboxList[(i >> 1) % 2][1], scaledBboxList[(i >> 2) % 2][2]]
            transformedPoint = np.add(transforms3d.quaternions.rotate_vector(point, rotationQuaternion), position).tolist()
            self.entities.extendByPoint(transformedPoint)

            lodPoint = [scaledLodBboxList[i % 2][0], scaledLodBboxList[(i >> 1) % 2][1], scaledLodBboxList[(i >> 2) % 2][2]]
            transformedLodPoint = np.add(transforms3d.quaternions.rotate_vector(lodPoint, rotationQuaternion), position).tolist()
            self.streaming.extendByPoint(transformedLodPoint)

    def isValid(self):
        return self.entities.isValid() and self.streaming.isValid()
