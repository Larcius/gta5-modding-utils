from re import Match
from typing import Any

import numpy as np
import transforms3d

import os
import re

from natsort import natsorted

from common.Box import Box
from common.Util import Util
from common.ymap.Flag import Flag


class StaticCollisionCreator:
    MAX_NUM_CHILDREN = -1  # 255
    ENTITIES_EXTENTS_MAX_DIAGONAL = 420
    IGNORE_PREVIOUS_FLAG_DISABLE_EMBEDED_COLLISION = True
    IGNORE_IF_SCALING_CLOSE_TO_IDENTITY = True

    templateObnContent: str

    def readTemplates(self):
        templateObnFile = open(os.path.join(os.path.dirname(__file__), "templates", "template.obn"), 'r')
        self.templateObnContent = templateObnFile.read()
        templateObnFile.close()

        # templateMapDataGroupsItemFile = open('template_map_data_groups_item.xml', 'r')
        # self.templateMapDataGroupsItemContent = templateMapDataGroupsItemFile.read()
        # templateMapDataGroupsItemFile.close()

        # templateManifestFile = open('template__manifest.ymf.xml', 'r')
        # self.templateManifestContent = templateManifestFile.read()
        # templateManifestFile.close()

    inputDir: str
    outputDir: str

    _colNumChildren: list[int]
    _colChildren: list[list[str]]
    _colChildFlags: list[list[str]]
    _entityIndex: int
    _clusters: Any

    @staticmethod
    def getRegExYmapCEntityDef():
        return '\\s*<Item type="CEntityDef">' + \
               '\\s*<archetypeName>([^<]+)</archetypeName>' + \
               '\\s*<flags value="([^"]+)"\\s*/>' + \
               '(?:\\s*<[^/].*>)' + \
               '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' + \
               '\\s*<rotation x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"\\s*/>' + \
               '\\s*<scaleXY value="([^"]+)"\\s*/>' + \
               '\\s*<scaleZ value="([^"]+)"\\s*/>' + \
               '(?:\\s*<[^/].*>)*' + \
               '\\s*<lodLevel>LODTYPES_DEPTH_(?:ORPHAN)?HD</lodLevel>' + \
               '(?:\\s*<[^/].*>)*' + \
               '\\s*</Item>'

    def __init__(self, inputDir: str, outputDir: str):
        self.inputDir = inputDir
        self.outputDir = outputDir

    def run(self):
        print("running static collision model creator...")
        self.readTemplates()
        self.createOutputDirs()
        self.processFiles()
        self.copyOthers()
        print("static collision model creator DONE")

    def createOutputDirs(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)
        os.mkdir(self.getOutputDirMaps())
        os.mkdir(self.getOutputDirCollisionModels())

        # if os.path.exists("cols"):
        #    shutil.copytree("cols", "generated/cols")
        # else:
        #    os.mkdir(self.outputDir + "/cols")

    def getOutputDirMaps(self):
        return os.path.join(self.outputDir, "maps")

    def getOutputDirCollisionModels(self):
        return os.path.join(self.outputDir, "cols")

    def createInitObn(self, bbox: Box, numChildren: int):
        # TODO using bbox might yield a too large bsphere
        # TODO not correct to use centeroid as center of gravity
        bsphere = bbox.getEnclosingSphere()

        return self.templateObnContent \
            .replace("${NUM_CHILDREN}", str(numChildren)) \
            .replace("${BSPHERE.CENTER.X}", Util.floatToStr(bsphere.center[0])) \
            .replace("${BSPHERE.CENTER.Y}", Util.floatToStr(bsphere.center[1])) \
            .replace("${BSPHERE.CENTER.Z}", Util.floatToStr(bsphere.center[2])) \
            .replace("${BSPHERE.RADIUS}", Util.floatToStr(bsphere.radius)) \
            .replace("${BBOX.MIN.X}", Util.floatToStr(bbox.min[0])) \
            .replace("${BBOX.MIN.Y}", Util.floatToStr(bbox.min[1])) \
            .replace("${BBOX.MIN.Z}", Util.floatToStr(bbox.min[2])) \
            .replace("${BBOX.MAX.X}", Util.floatToStr(bbox.max[0])) \
            .replace("${BBOX.MAX.Y}", Util.floatToStr(bbox.max[1])) \
            .replace("${BBOX.MAX.Z}", Util.floatToStr(bbox.max[2]))

    def transform(self, xyz: list[float], preTranslation: list[float], preTransformationMatrix, rotationQuaternion: list[float],
            scale: list[float], postTranslation: list[float]) -> list[float]:

        vec4PreTranslation = [xyz[0] + preTranslation[0], xyz[1] + preTranslation[1], xyz[2] + preTranslation[2], 1]
        vector = np.dot(vec4PreTranslation, preTransformationMatrix)
        transformed = transforms3d.quaternions.rotate_vector(vector, rotationQuaternion)
        transformed[0] *= scale[0]
        transformed[1] *= scale[1]
        transformed[2] *= scale[2]
        return [transformed[0] + postTranslation[0], transformed[1] + postTranslation[1], transformed[2] + postTranslation[2]]

    def transformStr(self, xyzStr: str, preTranslation: list[float], preTransformationMatrix, rotationQuaternion: list[float],
            scale: list[float], postTranslation: list[float]) -> str:

        xyzArr = xyzStr.split(" ")
        xyz = [float(xyzArr[0]), float(xyzArr[1]), float(xyzArr[2])]
        transformed = self.transform(xyz, preTranslation, preTransformationMatrix, rotationQuaternion, scale, postTranslation)
        return Util.floatToStr(transformed[0]) + " " + Util.floatToStr(transformed[1]) + " " + Util.floatToStr(transformed[2])

    def handleChildrenLine(self, line: str, childTransformationMatrix, position: list[float], rotationQuaternion: list[float], scale: list[float]) -> str:
        global geometryCenterOrig, negativeGeometryCenterNew

        match = re.match('(^		{)$', line)
        if match is not None:
            geometryCenterOrig = negativeGeometryCenterNew = None
            return line

        # this matches Radius for bsphere, not for Polygons
        match = re.match('(^			Radius )(\\S+)$', line)
        if match is not None:
            # TODO apply childTransformationMatrix to Radius first
            # transposed = np.transpose(childTransformationMatrix[:-1,:])
            # T, R, Z, S = transforms3d.affines.decompose44(transposed, )
            return match.group(1) + Util.floatToStr(float(match.group(2)) * max(scale)) + "\n"

        match = re.match('(^			(?:Centroid|CG) )(\\S+ \\S+ \\S+)$', line)
        if match is not None:
            transformed = self.transformStr(match.group(2), [0, 0, 0], childTransformationMatrix, rotationQuaternion, scale, position)
            return match.group(1) + transformed + "\n"

        # this matches Radius for Polygons, not for bsphere
        match = re.match('(^					Radius )(\\S+)$', line)
        if match is not None:
            # TODO apply childTransformationMatrix to Radius first
            # transposed = np.transpose(childTransformationMatrix[:-1,:])
            # T, R, Z, S = transforms3d.affines.decompose44(transposed, )
            return match.group(1) + Util.floatToStr(float(match.group(2)) * max(scale)) + "\n"

        match = re.match('(^\\s*GeometryCenter )(\\S+ \\S+ \\S+)( \\S+)$', line)
        if match is not None:
            geometryCenterSplitted = match.group(2).split(" ")
            geometryCenterOrig = [float(geometryCenterSplitted[0]), float(geometryCenterSplitted[1]), float(geometryCenterSplitted[2])]
            geometryCenterNew = self.transform(geometryCenterOrig, [0, 0, 0], childTransformationMatrix, rotationQuaternion, scale, [0, 0, 0])
            negativeGeometryCenterNew = [-geometryCenterNew[0], -geometryCenterNew[1], -geometryCenterNew[2]]

            geometryCenterNewStr = Util.floatToStr(geometryCenterNew[0] + position[0]) + " " + Util.floatToStr(geometryCenterNew[1] + position[1]) + \
                                   " " + Util.floatToStr(geometryCenterNew[2] + position[2])
            return match.group(1) + geometryCenterNewStr + match.group(3) + "\n"

        match = re.match('(^\\s*)([-+\\d.]+ [-+\\d.]+ [-+\\d.]+)$', line)
        if match is not None:
            transformed = self.transformStr(match.group(2), geometryCenterOrig, childTransformationMatrix, rotationQuaternion, scale, negativeGeometryCenterNew)
            return match.group(1) + transformed + "\n"

        return line

    def replaceBoundCapsule(self, match: Match) -> str:
        radius = float(match.group(2))
        heightHalfAndRadius = float(match.group(3))
        return """
		{
			Type BoundBVH""" + match.group(1) + """
			Polygons 1
			{
				Capsule 0
				{
					CenterTop 0
					CenterBottom 1
					Radius """ + Util.floatToStr(radius) + """
					MaterialIndex 0
				}
			}
			GeometryCenter 0.00000000 0.00000000 0.00000000 0.00250000
			Vertices 2
			{
				0.00000000 """ + Util.floatToStr(heightHalfAndRadius - radius) + """ 0.00000000
				0.00000000 """ + Util.floatToStr(-heightHalfAndRadius + radius) + """ 0.00000000
			}
			VertexColors null
			Materials 1
			{
				Material 0
				{
""" + match.group(4) + """
				}
			}
			MaterialColors null
			Margin 0.00500000
		}"""

    def replaceBoundCylinder(self, match: Match) -> str:
        radius = float(match.group(2))
        heightHalf = float(match.group(3))
        return """
		{
			Type BoundBVH""" + match.group(1) + """
			Polygons 1
			{
				Cylinder 0
				{
					CenterTop 0
					CenterBottom 1
					Radius """ + Util.floatToStr(radius) + """
					MaterialIndex 0
				}
			}
			GeometryCenter 0.00000000 0.00000000 0.00000000 0.00250000
			Vertices 2
			{
				0.00000000 """ + Util.floatToStr(heightHalf) + """ 0.00000000
				0.00000000 """ + Util.floatToStr(-heightHalf) + """ 0.00000000
			}
			VertexColors null
			Materials 1
			{
				Material 0
				{
""" + match.group(4) + """
				}
			}
			MaterialColors null
			Margin 0.00500000
		}"""

    def replaceBoundSphere(self, match) -> str:
        radius = float(match.group(2))
        return """
		{
			Type BoundBVH""" + match.group(1) + """
			Polygons 1
			{
				Sphere 0
				{
					Center 0
					Radius """ + Util.floatToStr(radius) + """
					MaterialIndex 0
				}
			}
			GeometryCenter 0.00000000 0.00000000 0.00000000 0.00250000
			Vertices 1
			{
				0.00000000 0.00000000 0.00000000
			}
			VertexColors null
			Materials 1
			{
				Material 0
				{
""" + match.group(3) + """
				}
			}
			MaterialColors null
			Margin 0.00500000
		}"""

    def replaceBoundBox(self, match) -> str:
        bboxMax = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
        bboxMin = [float(match.group(5)), float(match.group(6)), float(match.group(7))]
        return """
		{
			Type BoundBVH""" + match.group(1) + """
			Polygons 1
			{
				Box 0
				{
					Vertices 0 1 2 3
					MaterialIndex 0
				}
			}
			GeometryCenter 0.00000000 0.00000000 0.00000000 0.00250000
			Vertices 4
			{
				""" + Util.floatToStr(bboxMin[0]) + " " + Util.floatToStr(bboxMin[1]) + " " + Util.floatToStr(bboxMin[2]) + """
				""" + Util.floatToStr(bboxMax[0]) + " " + Util.floatToStr(bboxMin[1]) + " " + Util.floatToStr(bboxMax[2]) + """
				""" + Util.floatToStr(bboxMin[0]) + " " + Util.floatToStr(bboxMax[1]) + " " + Util.floatToStr(bboxMax[2]) + """
				""" + Util.floatToStr(bboxMax[0]) + " " + Util.floatToStr(bboxMax[1]) + " " + Util.floatToStr(bboxMin[2]) + """
			}
			VertexColors null
			Materials 1
			{
				Material 0
				{
""" + match.group(8) + """
				}
			}
			MaterialColors null
			Margin 0.00500000
		}"""

    def getColModelPathCandidate(self, entity: str) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "..", "resources", "models", entity.lower(), entity.lower() + ".bound")

    def isExistColModel(self, entity: str) -> bool:
        return os.path.exists(self.getColModelPathCandidate(entity))

    def shouldEntityBeUsedInStaticCol(self, entity: str, flags: int, scale: list[float]) -> bool:
        if not self.isExistColModel(entity):
            return False

        if not StaticCollisionCreator.IGNORE_PREVIOUS_FLAG_DISABLE_EMBEDED_COLLISION and flags & Flag.DISABLE_EMBEDDED_COLLISION:
            return False

        if StaticCollisionCreator.IGNORE_IF_SCALING_CLOSE_TO_IDENTITY and min(scale) >= 0.9 and max(scale) <= 1.1:
            return False

        return True

    def replaceYmapCEntityDef(self, match: Match) -> str:
        entity = match.group(1).lower()
        flags = int(match.group(2))

        scale = [float(match.group(10)), float(match.group(10)), float(match.group(11))]

        if not self.shouldEntityBeUsedInStaticCol(entity, flags, scale):
            if StaticCollisionCreator.IGNORE_PREVIOUS_FLAG_DISABLE_EMBEDED_COLLISION:
                flags &= ~Flag.DISABLE_EMBEDDED_COLLISION
                return re.sub('(?<=<flags value=")[^"]+("\\s*/>)', str(flags) + "\\g<1>", match.group(0), flags=re.M)
            else:
                return match.group(0)

        flags |= Flag.DISABLE_EMBEDDED_COLLISION

        position = [float(match.group(3)), float(match.group(4)), float(match.group(5))]
        rotationQuaternion = [float(match.group(9)), -float(match.group(6)), -float(match.group(7)), -float(match.group(8))]  # order is w, -x, -y, -z

        boundPath = self.getColModelPathCandidate(entity)

        boundFile = open(boundPath, 'r')
        boundContent = boundFile.read()
        boundFile.close()

        # special case: replace bounds of Type BoundCapsule by Type BoundBVH with Capsule as Polygons
        boundContent = re.sub('\\s*{' +
                              '\\s*Type BoundCapsule(' +
                              '\\s*Radius \\S+' +
                              '\\s*AABBMax (\\S+) (\\S+) \\S+' +
                              '\\s*AABBMin [^\n]+' +
                              '\\s*Centroid [^\n]+' +
                              '\\s*CG [^\n]+)' +
                              '\\s*CapsuleHalfHeight [^\n]+' +
                              '\\s*Margin \\S+' +
                              '\\s*Material' +
                              '\\s*{' +
                              '([^}]+)' +
                              '\\s*}' +
                              '\\s*}', self.replaceBoundCapsule, boundContent, flags=re.M)

        # special case: replace bounds of Type BoundCylinder by Type BoundBVH with Cylinder as Polygons
        boundContent = re.sub('\\s*{' +
                              '\\s*Type BoundCylinder(' +
                              '\\s*Radius \\S+' +
                              '\\s*AABBMax (\\S+) (\\S+) \\S+' +
                              '\\s*AABBMin [^\n]+' +
                              '\\s*Centroid [^\n]+' +
                              '\\s*CG [^\n]+)' +
                              '\\s*Margin \\S+' +
                              '\\s*Material' +
                              '\\s*{' +
                              '([^}]+)' +
                              '\\s*}' +
                              '\\s*}', self.replaceBoundCylinder, boundContent, flags=re.M)

        # special case: replace bounds of Type BoundSphere by Type BoundBVH with Sphere as Polygons
        boundContent = re.sub('\\s*{' +
                              '\\s*Type BoundSphere(' +
                              '\\s*Radius (\\S+)' +
                              '\\s*AABBMax [^\n]+' +
                              '\\s*AABBMin [^\n]+' +
                              '\\s*Centroid [^\n]+' +
                              '\\s*CG [^\n]+)' +
                              '\\s*Margin \\S+' +
                              '\\s*Material' +
                              '\\s*{' +
                              '([^}]+)' +
                              '\\s*}' +
                              '\\s*}', self.replaceBoundSphere, boundContent, flags=re.M)

        # special case: replace bounds of Type BoundBox by Type BoundBVH with Box as Polygons
        boundContent = re.sub('\\s*{' +
                              '\\s*Type BoundBox(' +
                              '\\s*Radius \\S+' +
                              '\\s*AABBMax (\\S+) (\\S+) (\\S+)' +
                              '\\s*AABBMin (\\S+) (\\S+) (\\S+)' +
                              '\\s*Centroid [^\n]+' +
                              '\\s*CG [^\n]+)' +
                              '\\s*Margin \\S+' +
                              '\\s*Material' +
                              '\\s*{' +
                              '([^}]+)' +
                              '\\s*}' +
                              '\\s*}', self.replaceBoundBox, boundContent, flags=re.M)

        mode = 0  # 0: no relevant block; 1: in Children block; 2: in ChildTransforms block; 3: in ChildFlags block
        for line in boundContent.splitlines(keepends=True):
            if mode != 0:
                if line.startswith("	}"):
                    mode = 0
                    continue
            elif line.startswith("	Children "):
                numChildren = int(line[10:-1])
                children = []
                mode = 1
                continue
            elif line.startswith("	ChildTransforms "):
                childTransforms = ""
                mode = 2
                continue
            elif line.startswith("	ChildFlags "):
                childFlags = []
                mode = 3
                continue

            if mode != 0 and line.startswith("	{"):
                continue

            if mode == 1:
                children.append(line)
            elif mode == 2:
                childTransforms += line
            elif mode == 3:
                childFlags.append(line)

        matrices = []
        for m in re.finditer('\\s*{' +
                             '\\s*(\\S+) (\\S+) (\\S+)' +
                             '\\s*(\\S+) (\\S+) (\\S+)' +
                             '\\s*(\\S+) (\\S+) (\\S+)' +
                             '\\s*(\\S+) (\\S+) (\\S+)' +
                             '\\s*}', childTransforms, flags=re.M):
            matrix = [[float(m.group(1)), float(m.group(2)), float(m.group(3))],
                [float(m.group(4)), float(m.group(5)), float(m.group(6))],
                [float(m.group(7)), float(m.group(8)), float(m.group(9))],
                [float(m.group(10)), float(m.group(11)), float(m.group(12))]]

            matrices.append(matrix)

        matrixIndex = -1
        for i in range(len(children)):
            if children[i] == "		{\n":
                matrixIndex += 1

            children[i] = self.handleChildrenLine(children[i], matrices[matrixIndex], position, rotationQuaternion, scale)

        cluster = self._clusters[self._entityIndex]
        self._entityIndex += 1

        self._colNumChildren[cluster] += numChildren
        self._colChildren[cluster].extend(children)
        self._colChildFlags[cluster].extend(childFlags)

        return re.sub('(?<=<flags value=")[^"]+("\\s*/>)', str(flags) + "\\g<1>", match.group(0), flags=re.M)

    def getVertex(self, children: list[str], index: int) -> list[float]:
        splitted = children[index][4:-1].split(" ")
        return [float(splitted[0]), float(splitted[1]), float(splitted[2])]

    def calculateBBox(self, children: list[str], translation: list[float], indexStartChild: int, indexStartPolygons: int, indexStartVertices: int,
            indexStartShrunk: int) -> Box:

        childBBox = Box.createReversedInfinityBox()

        i = indexStartPolygons
        while children[i] != "			}\n":
            line = children[i]
            i += 1

            if line.startswith("				Sphere"):
                # e.g. "\t\t\t\t\tCenter 12\n"
                centerVertexIndex = int(children[i + 1][11:-1])
                # e.g. "\t\t\t\t\tRadius 0.9043525\n"
                radius = float(children[i + 2][12:-1])

                i += 3

                center = self.getVertex(children, indexStartVertices + centerVertexIndex)

                childBBox.extendByPoint(np.subtract(center, [radius]).tolist())
                childBBox.extendByPoint(np.add(center, [radius]).tolist())
            elif line.startswith("				Capsule") or line.startswith("				Cylinder"):
                # e.g. "\t\t\t\t\tCenterTop 12\n"
                centerTopVertexIndex = int(children[i + 1][15:-1])
                # e.g. "\t\t\t\t\tCenterBottom 13\n"
                centerBottomVertexIndex = int(children[i + 2][18:-1])
                # e.g. "\t\t\t\t\tRadius 0.9043525\n"
                radius = float(children[i + 3][12:-1])

                i += 4

                centerTop = self.getVertex(children, indexStartVertices + centerTopVertexIndex)
                centerBottom = self.getVertex(children, indexStartVertices + centerBottomVertexIndex)

                # TODO for Cylinder this is not correct. However it is ensured that the calculated bbox is not smaller than the actual bbox
                childBBox.extendByPoint(np.subtract(centerTop, [radius]).tolist())
                childBBox.extendByPoint(np.subtract(centerBottom, [radius]).tolist())
                childBBox.extendByPoint(np.add(centerTop, [radius]).tolist())
                childBBox.extendByPoint(np.add(centerBottom, [radius]).tolist())

        i = indexStartVertices
        while children[i] != "			}\n":
            vertex = self.getVertex(children, i)
            i += 1
            childBBox.extendByPoint(vertex)

        i = indexStartShrunk
        while i >= 0 and children[i] != "			}\n":
            vertex = self.getVertex(children, i)
            i += 1
            childBBox.extendByPoint(vertex)

        childBBox = childBBox.getTranslated(translation)

        return childBBox

    def fixBBoxesAndBSpheres(self, children: list[str]) -> Box:
        bbox = Box.createReversedInfinityBox()

        translation = []
        indexStartChild = indexStartPolygons = indexStartVertices = indexStartShrunk = -1
        i = 0
        for line in children:
            i += 1
            if line.startswith("		{"):
                indexStartChild = i
            if line.startswith("			GeometryCenter "):
                xyzArr = line[18:-1].split(" ")
                translation = [float(xyzArr[0]), float(xyzArr[1]), float(xyzArr[2])]
            elif line.startswith("			Polygons "):
                indexStartPolygons = i + 1
            elif line.startswith("			Vertices "):
                indexStartVertices = i + 1
            elif line.startswith("			Shrunk "):
                indexStartShrunk = i + 1
            elif line.startswith("		}"):
                if indexStartChild == -1 or indexStartPolygons == -1 or indexStartVertices == -1:
                    open("dump.log", "w").writelines(children)
                    raise ValueError("could not find all indices but reached end of block in line " + str(i))

                childBBox = self.calculateBBox(children, translation, indexStartChild, indexStartPolygons, indexStartVertices, indexStartShrunk)

                # set bbox of child
                children[indexStartChild + 2] = "			AABBMax " + Util.floatToStr(childBBox.max[0]) + " " + Util.floatToStr(childBBox.max[1]) + \
                                                " " + Util.floatToStr(childBBox.max[2]) + "\n"
                children[indexStartChild + 3] = "			AABBMin " + Util.floatToStr(childBBox.min[0]) + " " + Util.floatToStr(childBBox.min[1]) + \
                                                " " + Util.floatToStr(childBBox.min[2]) + "\n"

                bbox.extendByPoint(childBBox.min)
                bbox.extendByPoint(childBBox.max)

        return bbox

    def processFiles(self):
        for mapFilename in natsorted(os.listdir(self.inputDir)):
            if not mapFilename.endswith(".ymap.xml"):
                continue

            self.processFile(mapFilename)

    def processFile(self, mapFilename: str):
        print("\tprocessing " + mapFilename)

        mapFile = open(os.path.join(self.inputDir, mapFilename), 'r')
        mapContent = mapFile.read()
        mapFile.close()

        # <!-- clustering
        pattern = re.compile('[\t ]*<Item type="CEntityDef">' +
                             '\\s*<archetypeName>([^<]+)</archetypeName>' +
                             '\\s*<flags value="([^"]+)"\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<position x="([^"]+)" y="([^"]+)" z="[^"]+"\\s*/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<scaleXY value="([^"]+)"/>' +
                             '\\s*<scaleZ value="([^"]+)"/>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*<lodLevel>LODTYPES_DEPTH_(?:ORPHAN)?HD</lodLevel>' +
                             '(?:\\s*<[^/].*>)*' +
                             '\\s*</Item>[\r\n]+')

        coords = []
        for matchobj in re.finditer(pattern, mapContent):
            if self.shouldEntityBeUsedInStaticCol(matchobj.group(1).lower(), int(matchobj.group(2)), [float(matchobj.group(5)), float(matchobj.group(6))]):
                coords.append([float(matchobj.group(3)), float(matchobj.group(4))])

        foundScolModel = len(coords) > 0

        if foundScolModel:
            self._clusters, slodEntitiesFurthestDistances = \
                Util.performClustering(coords, StaticCollisionCreator.MAX_NUM_CHILDREN, StaticCollisionCreator.ENTITIES_EXTENTS_MAX_DIAGONAL)

            numClusters = len(np.unique(self._clusters))

            self._colNumChildren = []
            self._colChildren = []
            self._colChildFlags = []
            for cluster in range(numClusters):
                self._colNumChildren.append(0)
                self._colChildren.append([])
                self._colChildFlags.append([])
        else:
            numClusters = -1
        # end of clustering -->

        self._entityIndex = 0
        mapContentNew = re.sub(StaticCollisionCreator.getRegExYmapCEntityDef(), self.replaceYmapCEntityDef, mapContent, flags=re.M)

        mapFileNew = open(os.path.join(self.getOutputDirMaps(), mapFilename), 'w')
        mapFileNew.write(mapContentNew)
        mapFileNew.close()

        if not foundScolModel:
            return

        mapName = mapFilename[:-9]

        for i in range(numClusters):
            numChildren = self._colNumChildren[i]
            children = self._colChildren[i]
            childFlags = self._colChildFlags[i]

            bbox = self.fixBBoxesAndBSpheres(children)

            # obnEmptyContentNew = createInitObn(bbox, 0) \
            # .replace("${CHILDREN.BOUNDS}\n", "") \
            # .replace("${CHILD_TRANSFORMS.MATRICES}\n", "") \
            # .replace("${CHILD_FLAGS.ITEMS}\n", "")
            # emptyObnFile = open(os.path.dirname(__file__) + "/generated/cols/" + mapName + "_" + str(i) + ".obn", 'w')
            # emptyObnFile.write(obnEmptyContentNew)
            # emptyObnFile.close()

            colFilename = "hi@" + mapName
            if numClusters > 1:
                colFilename += "_" + str(i)
            colFilename += ".obn"

            obnFileNew = open(os.path.join(self.getOutputDirCollisionModels(), colFilename), 'w')

            i += 1

            obnContentNew = self.createInitObn(bbox, numChildren)
            for line in obnContentNew.splitlines(keepends=True):
                if line == "${CHILDREN.BOUNDS}\n":
                    obnFileNew.writelines(children)
                elif line == "${CHILD_TRANSFORMS.MATRICES}\n":
                    for indexMatrix in range(numChildren):
                        obnFileNew.write("""		Matrix """ + str(indexMatrix) + """
		{
			1.00000000 0.00000000 0.00000000
			0.00000000 1.00000000 0.00000000
			0.00000000 0.00000000 1.00000000
			0.00000000 0.00000000 0.00000000
		}
""")
                elif line == "${CHILD_FLAGS.ITEMS}\n":
                    obnFileNew.writelines(childFlags)
                else:
                    obnFileNew.write(line)
            obnFileNew.close()

        # colItems = ""
        # for i in range(numClusters):
        #     colItems += "		<Item>hi@" + mapName + "_" + str(i) + "</Item>\n"
        #
        #	mapDataGroupsItems += templateMapDataGroupsItemContent.replace("${MAP.NAME}", mapName).replace("${MAP.BOUNDS.ITEMS}\n", colItems)
        #
        # manifestFile = open(os.path.dirname(__file__) + "/generated/cols/_manifest.ymf.xml", 'w')
        # for line in templateManifestContent.splitlines(keepends = True):
        #	if line == "${MAP_DATA_GROUPS}\n":
        #		manifestFile.writelines(mapDataGroupsItems)
        #	else:
        #		manifestFile.write(line)
        # manifestFile.close()

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.getOutputDirMaps(), lambda filename: not filename.endswith(".ymap.xml"))
