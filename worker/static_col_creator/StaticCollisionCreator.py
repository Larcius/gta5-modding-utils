import copy
from re import Match
from typing import Any

import numpy as np

import os
import re

from natsort import natsorted

from common.Util import Util
from common.ymap.Flag import Flag
from worker.static_col_creator.BoundComposite import BoundComposite


class StaticCollisionCreator:
    MAX_NUM_CHILDREN = -1
    ENTITIES_EXTENTS_MAX_DIAGONAL = 400
    IGNORE_PREVIOUS_FLAG_DISABLE_EMBEDED_COLLISION = True
    IGNORE_IF_SCALING_CLOSE_TO_IDENTITY = True
    IGNORE_IF_SCALING_CLOSE_TO_IDENTITY_TOLERANCE = 1.05

    inputDir: str
    outputDir: str

    _entityColModels: dict[str, BoundComposite]

    _colChildren: list[BoundComposite]
    _entityIndex: int
    _clusters: Any

    @staticmethod
    def getRegExYmapCEntityDef():
        return '\\s*<Item type="CEntityDef">' + \
               '\\s*<archetypeName>([^<]+)</archetypeName>' + \
               '\\s*<flags value="([^"]+)"\\s*/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' + \
               '\\s*<rotation x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"\\s*/>' + \
               '\\s*<scaleXY value="([^"]+)"\\s*/>' + \
               '\\s*<scaleZ value="([^"]+)"\\s*/>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*<lodLevel>LODTYPES_DEPTH_(?:ORPHAN)?HD</lodLevel>' + \
               '(?:\\s*<[^/].*>)*?' + \
               '\\s*</Item>'

    def __init__(self, inputDir: str, outputDir: str):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self._entityColModels = {}

    def run(self):
        print("running static collision model creator...")
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
					""" + match.group(4).replace("\n", "\n\t") + """
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
					""" + match.group(4).replace("\n", "\n\t") + """
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
					""" + match.group(3).replace("\n", "\n\t") + """
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
					""" + match.group(8).replace("\n", "\n\t") + """
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

        if StaticCollisionCreator.IGNORE_IF_SCALING_CLOSE_TO_IDENTITY \
                and min(scale) >= 1 / StaticCollisionCreator.IGNORE_IF_SCALING_CLOSE_TO_IDENTITY_TOLERANCE \
                and max(scale) <= StaticCollisionCreator.IGNORE_IF_SCALING_CLOSE_TO_IDENTITY_TOLERANCE:
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

        boundComposite = self.getEntityColModel(entity)

        boundComposite.transform(rotationQuaternion, scale, position)

        cluster = self._clusters[self._entityIndex]
        self._entityIndex += 1

        self.mergeColChildren(cluster, boundComposite)

        return re.sub('(?<=<flags value=")[^"]+("\\s*/>)', str(flags) + "\\g<1>", match.group(0), flags=re.M)

    def getEntityColModel(self, entity: str) -> BoundComposite:
        if entity not in self._entityColModels:
            boundPath = self.getColModelPathCandidate(entity)
            boundContent = Util.readFile(boundPath)

            boundContent = self.convertToBoundComposite(boundContent)
            boundContent = self.convertToBoundBVH(boundContent)

            self._entityColModels[entity] = BoundComposite.parse(boundContent)

        return copy.deepcopy(self._entityColModels[entity])

    def convertToBoundComposite(self, boundContent: str) -> str:
        if boundContent.startswith("Version 43 31\n{\n\tType BoundComposite\n"):
            return boundContent

        start = boundContent.index("{") + 1
        end = boundContent.rindex("}")
        boundMiddle = boundContent[start:end].strip().replace("\n", "\n\t\t")
        return """Version 43 31
{
	Type BoundComposite
	Radius 0.00000000
	AABBMax 0.00000000 0.00000000 0.00000000
	AABBMin 0.00000000 0.00000000 0.00000000
	Centroid 0.00000000 0.00000000 0.00000000
	CG 0.00000000 0.00000000 0.00000000
	Children 1
	{
		phBound
		{
			""" + boundMiddle + """
		}
	}
	ChildTransforms 1
	{
		Matrix 0
		{
			1.00000000 0.00000000 0.00000000
			0.00000000 1.00000000 0.00000000
			0.00000000 0.00000000 1.00000000
			0.00000000 0.00000000 0.00000000
		}
	}
	ChildFlags 1
	{
		Item
		{
			Flags1 MAP_WEAPON MAP_DYNAMIC MAP_ANIMAL MAP_COVER MAP_VEHICLE
			Flags2 VEHICLE_NOT_BVH VEHICLE_BVH PED RAGDOLL ANIMAL ANIMAL_RAGDOLL OBJECT PLANT PROJECTILE EXPLOSION FORKLIFT_FORKS TEST_WEAPON TEST_CAMERA TEST_AI TEST_SCRIPT TEST_VEHICLE_WHEEL GLASS
		}
	}
}
"""

    def convertToBoundBVH(self, boundContent):
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
                              '\\s*([^}]+?)' +
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
                              '\\s*([^}]+?)' +
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
                              '\\s*([^}]+?)' +
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
                              '\\s*([^}]+?)' +
                              '\\s*}' +
                              '\\s*}', self.replaceBoundBox, boundContent, flags=re.M)
        return boundContent

    def mergeColChildren(self, cluster: int, boundComposite: BoundComposite):
        self._colChildren[cluster].merge(boundComposite)

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
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<scaleXY value="([^"]+)"/>' +
                             '\\s*<scaleZ value="([^"]+)"/>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*<lodLevel>LODTYPES_DEPTH_(?:ORPHAN)?HD</lodLevel>' +
                             '(?:\\s*<[^/].*>)*?' +
                             '\\s*</Item>[\r\n]+')

        coords = []
        for matchobj in re.finditer(pattern, mapContent):
            if self.shouldEntityBeUsedInStaticCol(matchobj.group(1).lower(), int(matchobj.group(2)), [float(matchobj.group(6)), float(matchobj.group(7))]):
                coords.append([float(matchobj.group(3)), float(matchobj.group(4)), float(matchobj.group(5))])

        foundScolModel = len(coords) > 0

        if foundScolModel:
            self._clusters, slodEntitiesFurthestDistances = \
                Util.performClustering(coords, StaticCollisionCreator.MAX_NUM_CHILDREN, StaticCollisionCreator.ENTITIES_EXTENTS_MAX_DIAGONAL)

            numClusters = len(np.unique(self._clusters))

            self._colChildren = []
            for cluster in range(numClusters):
                self._colChildren.append(BoundComposite([]))
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
            # obnEmptyContentNew = createInitObn(bbox, 0) \
            # .replace("${CHILDREN.BOUNDS}\n", "") \
            # .replace("${CHILD_TRANSFORMS.MATRICES}\n", "") \
            # .replace("${CHILD_FLAGS.ITEMS}\n", "")
            # emptyObnFile = open(os.path.dirname(__file__) + "/generated/cols/" + mapName + "_" + str(i) + ".obn", 'w')
            # emptyObnFile.write(obnEmptyContentNew)
            # emptyObnFile.close()

            colDefaultFilename = mapName
            if numClusters > 1:
                colDefaultFilename += "_" + str(i)
            colDefaultFilename += ".obn"

            boundDefault, boundMa, boundHi = self._colChildren[i].splitIntoDefaultMaHi()

            for mode in range(3):
                if mode == 0:
                    colFilename = "hi@" + colDefaultFilename
                    bound = boundHi
                elif mode == 1:
                    colFilename = "ma@" + colDefaultFilename
                    bound = boundMa
                else:
                    colFilename = colDefaultFilename
                    bound = boundDefault

                bound.writeToFile(os.path.join(self.getOutputDirCollisionModels(), colFilename))

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
