from re import Match
from typing import Any, IO, Optional

import numpy as np
import transforms3d
from natsort import natsorted

import os
import re
import shutil
import math

from common.Box import Box
from common.Sphere import Sphere
from common.Util import Util
from common.texture.UV import UV
from common.texture.UVMap import UVMap
from common.ymap.ContentFlag import ContentFlag
from common.ymap.EntityItem import EntityItem
from common.ymap.Flag import Flag
from common.ymap.LodLevel import LodLevel
from common.ymap.PriorityLevel import PriorityLevel
from common.ymap.Ymap import Ymap
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class LodMapCreator:
    inputDir: str
    outputDir: str

    prefix: str
    bundlePrefixes: list[str]

    contentTemplateYtypItem: str
    contentTemplateMesh: str
    contentTemplateMeshAabb: str
    contentTemplateMeshGeometry: str
    contentTemplateOdr: str
    contentTemplateOdrShaderTreeLod: str
    contentTemplateOdrShaderTreeLod2: str
    contentTemplateEntitySlod: str
    contentTemplateSlod2Map: str

    ytypItems: dict[str, YtypItem]
    slodCandidates: dict[str, UVMap]

    _foundSlodModel: bool
    _slodEntitiesFurthestDistances: list[float]
    _entitiesForSlod: list[list[EntityItem]]
    _clusters: Any
    _slodIndex: int
    _index: int
    _translation: list[float]

    slodYtypItems: IO

    def prepareSlodCandidates(self):
        # TODO provide more slod models:
        # prop_rio_del_01

        slodCandidates = {
            # trees
            "Prop_Tree_LFicus_02": UVMap("trees2", UV(0 / 2, 0 / 2), UV(1 / 2, 1 / 2), UV(0 / 2, 0 / 2), UV(1 / 2, 1 / 2)),
            "Prop_Tree_LFicus_03": UVMap("trees2", UV(1 / 2, 0 / 2), UV(2 / 2, 1 / 2), UV(1 / 2, 0 / 2), UV(2 / 2, 1 / 2)),
            "Prop_Tree_LFicus_05": UVMap("trees2", UV(0 / 2, 1 / 2), UV(1 / 2, 2 / 2), UV(0 / 2, 1 / 2), UV(1 / 2, 2 / 2)),
            "Prop_Tree_LFicus_06": UVMap("trees2", UV(1 / 2, 1 / 2), UV(2 / 2, 2 / 2), UV(1 / 2, 1 / 2), UV(2 / 2, 2 / 2)),
            "Prop_S_Pine_Dead_01": UVMap("trees", UV(10 / 16, 6 / 16), UV(12 / 16, 11 / 16), UV(0 / 16, 12 / 16), UV(3 / 16, 14 / 16)),
            "Prop_Tree_Birch_01": UVMap("trees", UV(8 / 16, 6 / 16), UV(10 / 16, 10 / 16), UV(0 / 16, 3 / 16), UV(3 / 16, 6 / 16)),
            "Prop_Tree_Birch_02": UVMap("trees", UV(6 / 16, 3 / 16), UV(9 / 16, 6 / 16), UV(3 / 16, 3 / 16), UV(6 / 16, 6 / 16)),
            "Prop_Tree_Birch_03": UVMap("trees", UV(4 / 16, 3 / 16), UV(6 / 16, 6 / 16)),
            "Prop_Tree_Birch_03b": UVMap("trees", UV(2 / 16, 3 / 16), UV(4 / 16, 5 / 16)),
            "Prop_Tree_Birch_04": UVMap("trees", UV(9 / 16, 3 / 16), UV(12 / 16, 6 / 16), UV(6 / 16, 3 / 16), UV(9 / 16, 6 / 16)),
            "Prop_Tree_Cedar_02": UVMap("trees", UV(4 / 16, 11 / 16), UV(6 / 16, 16 / 16), UV(6 / 16, 6 / 16), UV(9 / 16, 9 / 16)),
            "Prop_Tree_Cedar_03": UVMap("trees", UV(6 / 16, 10 / 16), UV(8 / 16, 16 / 16), UV(9 / 16, 6 / 16), UV(12 / 16, 9 / 16)),
            "Prop_Tree_Cedar_04": UVMap("trees", UV(8 / 16, 10 / 16), UV(10 / 16, 16 / 16), UV(12 / 16, 6 / 16), UV(15 / 16, 9 / 16)),
            "Prop_Tree_Cedar_S_01": UVMap("trees", UV(6 / 16, 6 / 16), UV(8 / 16, 10 / 16), UV(0 / 16, 6 / 16), UV(3 / 16, 9 / 16)),
            "Prop_Tree_Cedar_S_04": UVMap("trees", UV(3 / 16, 8 / 16), UV(4 / 16, 12 / 16), UV(3 / 16, 6 / 16), UV(3 / 16, 6 / 16)),
            "Prop_Tree_Cedar_S_05": UVMap("trees", UV(3 / 16, 8 / 16), UV(4 / 16, 12 / 16)),
            "Prop_Tree_Cedar_S_06": UVMap("trees", UV(6 / 16, 0 / 16), UV(7 / 16, 3 / 16)),
            "Prop_Tree_Cypress_01": UVMap("trees", UV(4 / 16, 6 / 16), UV(6 / 16, 11 / 16)),
            "Prop_Tree_Eng_Oak_01": UVMap("trees", UV(10 / 16, 0 / 16), UV(13 / 16, 3 / 16), UV(9 / 16, 0 / 16), UV(12 / 16, 3 / 16)),
            "Prop_Tree_Eucalip_01": UVMap("trees", UV(0 / 16, 8 / 16), UV(3 / 16, 12 / 16), UV(9 / 16, 3 / 16), UV(12 / 16, 6 / 16)),
            "Prop_Tree_Fallen_Pine_01": UVMap("trees", UV(12 / 16, 3 / 16), UV(14 / 16, 8 / 16), UV(0 / 16, 14 / 16), UV(3 / 16, 16 / 16)),
            "Prop_Tree_Jacada_01": UVMap("trees", UV(0 / 16, 0 / 16), UV(3 / 16, 3 / 16), UV(0 / 16, 0 / 16), UV(3 / 16, 3 / 16)),
            "Prop_Tree_Jacada_02": UVMap("trees", UV(3 / 16, 0 / 16), UV(6 / 16, 3 / 16), UV(3 / 16, 0 / 16), UV(6 / 16, 3 / 16)),
            "Prop_Tree_Maple_02": UVMap("trees", UV(0 / 16, 3 / 16), UV(2 / 16, 5 / 16)),
            "Prop_Tree_Maple_03": UVMap("trees", UV(0 / 16, 5 / 16), UV(2 / 16, 8 / 16)),
            "Prop_Tree_Oak_01": UVMap("trees", UV(13 / 16, 0 / 16), UV(16 / 16, 3 / 16), UV(12 / 16, 0 / 16), UV(15 / 16, 3 / 16)),
            "Prop_Tree_Olive_01": UVMap("trees", UV(7 / 16, 0 / 16), UV(10 / 16, 3 / 16), UV(6 / 16, 0 / 16), UV(9 / 16, 3 / 16)),
            "Prop_Tree_Pine_01": UVMap("trees", UV(0 / 16, 12 / 16), UV(2 / 16, 16 / 16), UV(0 / 16, 9 / 16), UV(3 / 16, 12 / 16)),
            "Prop_Tree_Pine_02": UVMap("trees", UV(2 / 16, 12 / 16), UV(4 / 16, 16 / 16), UV(3 / 16, 9 / 16), UV(6 / 16, 12 / 16)),
            "Prop_W_R_Cedar_01": UVMap("trees", UV(10 / 16, 11 / 16), UV(12 / 16, 16 / 16), UV(6 / 16, 9 / 16), UV(9 / 16, 12 / 16)),
            "Prop_W_R_Cedar_Dead": UVMap("trees", UV(14 / 16, 3 / 16), UV(16 / 16, 8 / 16), UV(3 / 16, 12 / 16), UV(6 / 16, 15 / 16)),
            "TEST_Tree_Cedar_Trunk_001": UVMap("trees", UV(12 / 16, 8 / 16), UV(14 / 16, 16 / 16), UV(9 / 16, 9 / 16), UV(12 / 16, 12 / 16)),
            "TEST_Tree_Forest_Trunk_01": UVMap("trees", UV(14 / 16, 8 / 16), UV(16 / 16, 16 / 16), UV(12 / 16, 9 / 16), UV(15 / 16, 12 / 16)),
            "TEST_Tree_Forest_Trunk_04": UVMap("trees", UV(2 / 16, 5 / 16), UV(4 / 16, 8 / 16), UV(12 / 16, 3 / 16), UV(15 / 16, 6 / 16)),
            # bushes
            "Prop_Bush_Lrg_04b": UVMap("bushes", UV(0.5, 0), UV(1, 0.5), UV(0.5, 0), UV(1, 0.5)),
            "Prop_Bush_Lrg_04c": UVMap("bushes", UV(0, 0.5), UV(0.5, 1), UV(0, 0.5), UV(0.5, 1)),
            "Prop_Bush_Lrg_04d": UVMap("bushes", UV(0.5, 0.5), UV(1, 1), UV(0.5, 0.5), UV(1, 1)),
            # palms
            "Prop_Palm_Sm_01e": UVMap("palms", UV(0 / 4, 0 / 4), UV(1 / 4, 2 / 4)),
            "Prop_Palm_Fan_02_b": UVMap("palms", UV(0 / 4, 2 / 4), UV(1 / 4, 4 / 4), UV(0, 0), UV(0.5, 0.5), 0.23692810457),
            "Prop_Palm_Fan_03_c": UVMap("palms", UV(1 / 4, 0 / 4), UV(2 / 4, 4 / 4), UV(0.5, 0), UV(1, 0.5), 0.14356435643),
            "Prop_Palm_Fan_03_d": UVMap("palms", UV(2 / 4, 0 / 4), UV(3 / 4, 4 / 4), UV(0, 0.5), UV(0.5, 1), 0.13046937151),
            "Prop_Palm_Huge_01a": UVMap("palms", UV(3 / 4, 0 / 4), UV(4 / 4, 4 / 4), UV(0.5, 0.5), UV(1, 1), 0.09644268774),
        }
        # add other Props that should use the same UV mapping
        slodCandidates["Prop_Palm_Sm_01d"] = slodCandidates["Prop_Palm_Sm_01f"] = slodCandidates["Prop_Palm_Med_01a"] = slodCandidates["Prop_Palm_Med_01b"] = \
            slodCandidates["Prop_Palm_Med_01c"] = slodCandidates["Prop_Palm_Sm_01e"]
        slodCandidates["Prop_Fan_Palm_01a"] = slodCandidates["Prop_Palm_Fan_02_a"] = slodCandidates["Prop_Palm_Sm_01a"] = slodCandidates["Prop_Palm_Fan_04_a"] = \
            slodCandidates["Prop_Palm_Fan_04_b"] = slodCandidates["Prop_Palm_Fan_02_b"]
        slodCandidates["Prop_Palm_Fan_03_a"] = slodCandidates["Prop_Palm_Fan_03_b"] = slodCandidates["Prop_Palm_Fan_03_c_graff"] = slodCandidates[
            "Prop_Palm_Fan_04_c"] = \
            slodCandidates["Prop_Palm_Fan_03_c"]
        slodCandidates["Prop_Palm_Med_01d"] = slodCandidates["Prop_Palm_Fan_03_d_Graff"] = slodCandidates["Prop_Palm_Fan_04_d"] = \
            slodCandidates["Prop_Palm_Fan_03_d"]
        slodCandidates["Prop_Palm_Huge_01b"] = \
            slodCandidates["Prop_Palm_Huge_01a"]

        self.slodCandidates = slodCandidates

    LOD_DISTANCE = 750  # the actual lodDistance for a lod entity is obtained by adding the furthest distance of all lod entities sharing the same slod model
    SLOD_DISTANCE = 1500  # somehow arbitrary and not that important because SLOD1 and SLOD2 are actually the same models.
    # however, reducing this results in smaller streaming extents of LOD/SLOD1 maps
    SLOD2_DISTANCE = 3000  # using 3000 because max height in game is 2600 and therefore until that height (plus a bit to allow slight xy offset)
    # a model with xy plane is needed so that objects don't vanish when above (SLOD2 models do contain such a xy plane)
    SLOD3_DISTANCE = 15000  # that seems to be the default value from Rockstars (in fact the whole map is not that large anyway)

    NUM_CHILDREN_MAX_VALUE = 255  # TODO confirm following claim: must be <= 255 since numChildren is of size 1 byte
    ENTITIES_EXTENTS_MAX_DIAGONAL = 420

    # only entities with a lodDistance (according to hd entity) greater or equal this value are considered for SLOD1 to 3 model
    unitBox = Box([-0.5] * 3, [0.5] * 3)
    unitSphere = Sphere([0] * 3, 1)
    MIN_HD_LOD_DISTANCE_FOR_SLOD1 = Util.calculateLodDistance(unitBox, unitSphere, [4] * 3, True)  # 180
    MIN_HD_LOD_DISTANCE_FOR_SLOD2 = Util.calculateLodDistance(unitBox, unitSphere, [8] * 3, True)  # 240
    MIN_HD_LOD_DISTANCE_FOR_SLOD3 = Util.calculateLodDistance(unitBox, unitSphere, [13] * 3, True)  # 290

    TEXTURE_UV_EPS = 1 / 512

    def __init__(self, inputDir: str, outputDir: str, prefix: str):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix

    def run(self):
        print("running lod map creator...")
        self.determinePrefixBundles()
        self.readTemplates()
        self.prepareSlodCandidates()
        self.createOutputDir()
        self.readYtypItems()
        self.processFiles()
        self.fixMapExtents()
        self.copyOthers()
        print("lod map creator DONE")

    def getYtypName(self) -> str:
        return self.prefix + "_slod"

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)
        os.mkdir(self.getOutputDirMaps())
        os.mkdir(self.getOutputDirMeshes())
        os.mkdir(self.getOutputDirModels())

    def getOutputDirMaps(self) -> str:
        return os.path.join(self.outputDir, "maps")

    def getOutputDirModels(self) -> str:
        return os.path.join(self.outputDir, "slod")

    def getOutputDirMeshes(self) -> str:
        return os.path.join(self.outputDir, "_meshes")

    def readTemplates(self):
        templatesDir = os.path.join(os.path.dirname(__file__), "templates")

        f = open(os.path.join(templatesDir, "template_lod_ytyp_item.xml"), 'r')
        self.contentTemplateYtypItem = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_slod_entity.ymap.xml"), 'r')
        self.contentTemplateEntitySlod = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_slod2.ymap.xml"), 'r')
        self.contentTemplateSlod2Map = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_slod.mesh"), 'r')
        self.contentTemplateMesh = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_aabb.mesh.part"), 'r')
        self.contentTemplateMeshAabb = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_geometry.mesh.part"), 'r')
        self.contentTemplateMeshGeometry = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_slod.odr"), 'r')
        self.contentTemplateOdr = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_shader_tree_lod.odr.part"), 'r')
        self.contentTemplateOdrShaderTreeLod = f.read()
        f.close()

        f = open(os.path.join(templatesDir, "template_shader_tree_lod2.odr.part"), 'r')
        self.contentTemplateOdrShaderTreeLod2 = f.read()
        f.close()

    def readYtypItems(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "ytyp"))

    def replaceName(self, content: str, name: str) -> str:
        return re.sub('(?<=<CMapData>)(\\s*<name>)[^<]+(?=</name>)', "\\g<1>" + name, content)

    def replaceParent(self, content: str, parent: Optional[str]) -> str:
        if parent == "" or parent is None:
            newParent = "<parent/>"
        else:
            newParent = "<parent>" + parent + "</parent>"
        return re.sub('<parent\\s*(?:/>|>[^<]*</parent>)', newParent, content, flags=re.M)

    def replaceFlagsAndContentFlags(self, content: str, flags: int, contentFlags: int) -> str:
        # TODO deal with existing flags, e.g. "Scripted (1)"
        return re.sub(
            '(<flags\\s+value=")[^"]*("\\s*/>\\s*<contentFlags\\s+value=")[^"]*("\\s*/>)',
            "\\g<1>" + str(flags) + "\\g<2>" + str(contentFlags) + "\\g<3>", content
        )

    def getLod(self, hdAsset: str) -> Optional[str]:
        if hdAsset + "_LOD" in self.ytypItems:
            return hdAsset + "_LOD"
        else:
            return None

    def replLod(self, matchobj: Match) -> str:
        hdAsset = matchobj.group(2)

        lodAsset = self.getLod(hdAsset)

        if lodAsset is None:
            return ""

        flags = int(matchobj.group(4))
        flags |= Flag.FLAGS_HD_DEFAULT  # add default flags
        flags &= ~Flag.DISABLE_EMBEDDED_COLLISION  # remove flag "disable embedded collision"
        flags &= ~Flag.UNKNOWN_3  # remove flag 8
        flags &= ~Flag.STATIC_ENTITY  # remove flag "static entity"

        childLodDistanceStr = matchobj.group(16)
        childLodDistance = float(childLodDistanceStr)

        # TODO clean way to determine if this should be a SLOD1 entity
        if self._foundSlodModel and hdAsset in self.slodCandidates and childLodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD1:
            position = [float(matchobj.group(6)), float(matchobj.group(7)), float(matchobj.group(8))]
            rotation = [float(matchobj.group(12)), -float(matchobj.group(9)), -float(matchobj.group(10)), -float(matchobj.group(11))]  # order is w, -x, -y, -z
            scale = [float(matchobj.group(13)), float(matchobj.group(13)), float(matchobj.group(14))]

            slodCluster = self._clusters[self._slodIndex]

            self._entitiesForSlod[slodCluster].append(EntityItem(hdAsset, position, scale, rotation, childLodDistance))

            lodParentIndex = slodCluster

            lodDistance = self.getLodDistanceForSlodModel(slodCluster)

            self._slodIndex += 1
        else:
            # TODO rather use LOD_DISTANCE as a maximum and consider lodDistance of HD entity as well, e.g. lodDistance = min(LOD_DISTANCE, 2 * hdLodDistance)
            lodDistance = LodMapCreator.LOD_DISTANCE
            lodParentIndex = -1

        return matchobj.group(1) + lodAsset + matchobj.group(3) + str(flags) + matchobj.group(5) + ' value="' + str(lodParentIndex) + '"' + \
               matchobj.group(15) + Util.floatToStr(lodDistance) + matchobj.group(17) + childLodDistanceStr + matchobj.group(18) + LodLevel.LOD + \
               matchobj.group(19) + "1" + matchobj.group(20) + PriorityLevel.REQUIRED + matchobj.group(21)

    def replParentIndex(self, matchobj: Match) -> str:
        lodAsset = self.getLod(matchobj.group(2))

        if lodAsset is None:
            parentIndex = "-1"
        else:
            parentIndex = str(self._index)
            self._index += 1

        return matchobj.group(1) + parentIndex + matchobj.group(3)

    def replacePlaceholders(self, template: str, name: str, bbox: Box, hdDistance: float, lodDistance: float) -> str:
        bsphere = bbox.getEnclosingSphere()

        return template \
            .replace("${NAME}", name) \
            .replace("${TEXTURE_DICTIONARY}", self.getYtypName()) \
            .replace("${LOD_DISTANCE}", Util.floatToStr(lodDistance)) \
            .replace("${HD_TEXTURE_DISTANCE}", Util.floatToStr(hdDistance)) \
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

    def replTranslateVertex(self, match: Match) -> str:
        vertex = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        newVertex = np.subtract(vertex, self._translation).tolist()
        return Util.floatToStr(newVertex[0]) + " " + Util.floatToStr(newVertex[1]) + " " + Util.floatToStr(newVertex[2])

    def createTextureUvWithEps(self, uvMin: Optional[UV], uvMax: Optional[UV]) -> (Optional[UV], Optional[UV]):
        if uvMin is None or uvMax is None:
            return uvMin, uvMax

        minUvEps = UV(uvMin.u, uvMin.v)
        maxUvEps = UV(uvMax.u, uvMax.v)
        if minUvEps.u < maxUvEps.u:
            minUvEps.u += LodMapCreator.TEXTURE_UV_EPS
        else:
            maxUvEps.u += LodMapCreator.TEXTURE_UV_EPS

        if minUvEps.v < maxUvEps.v:
            minUvEps.v += LodMapCreator.TEXTURE_UV_EPS
        else:
            maxUvEps.v += LodMapCreator.TEXTURE_UV_EPS

        return minUvEps, maxUvEps

    def createIndices(self, numRectangles: int) -> str:
        indices = ""
        for i in range(numRectangles):
            if (i * 6) % 15 == 0:
                if i > 0:
                    indices += "\n"
                indices += "				"
            else:
                indices += " "
            indices += str(i * 4) + " " + str(i * 4 + 1) + " " + str(i * 4 + 2)

            if (i * 6 + 3) % 15 == 0:
                indices += "\n				"
            else:
                indices += " "
            indices += str(i * 4 + 2) + " " + str(i * 4 + 3) + " " + str(i * 4)

        return indices

    def createAabb(self, bbox: Box) -> str:
        return self.contentTemplateMeshAabb \
            .replace("${BBOX.MIN.X}", Util.floatToStr(bbox.min[0])) \
            .replace("${BBOX.MIN.Y}", Util.floatToStr(bbox.min[1])) \
            .replace("${BBOX.MIN.Z}", Util.floatToStr(bbox.min[2])) \
            .replace("${BBOX.MAX.X}", Util.floatToStr(bbox.max[0])) \
            .replace("${BBOX.MAX.Y}", Util.floatToStr(bbox.max[1])) \
            .replace("${BBOX.MAX.Z}", Util.floatToStr(bbox.max[2]))

    def createSlodModel(self, nameWithoutSlodLevel: str, slodLevel: int, entities: list[EntityItem]) -> (Optional[str], Optional[list[float]]):
        name = nameWithoutSlodLevel + str(slodLevel)

        normalAndColorsFront = " / 0.00000000 -1.00000000 0.00000000 / 255 29 0 255 / 0 255 0 0 / "
        normalAndColorsTop = " / 0.00000000 0.00000000 1.00000000 / 255 29 0 255 / 0 0 0 0 / "
        verticesFront = {}
        verticesTop = {}
        numFrontPlanes = {}
        numTopPlanes = {}
        bbox = {}
        numSkippedSlodEntities = 0
        for entity in entities:
            # if slodLevel == 1 and entity.lodDistance < MIN_HD_LOD_DISTANCE_FOR_SLOD1:
            #    numSkippedSlodEntities += 1
            #    continue
            if slodLevel == 2 and entity.lodDistance < LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD2:
                numSkippedSlodEntities += 1
                continue
            if slodLevel == 3 and entity.lodDistance < LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD3:
                numSkippedSlodEntities += 1
                continue

            uvMap = self.slodCandidates[entity.archetypeName]

            diffuseSampler = uvMap.getDiffuseSampler()

            if diffuseSampler not in verticesFront:
                verticesFront[diffuseSampler] = ""
                verticesTop[diffuseSampler] = ""
                numFrontPlanes[diffuseSampler] = 0
                numTopPlanes[diffuseSampler] = 0
                bbox[diffuseSampler] = Box.createReversedInfinityBox()

            bboxEntity = self.ytypItems[entity.archetypeName].boundingBox

            uvFrontMin, uvFrontMax = self.createTextureUvWithEps(uvMap.frontMin, uvMap.frontMax)
            uvTopMin, uvTopMax = self.createTextureUvWithEps(uvMap.topMin, uvMap.topMax)
            textureMinStr = [Util.floatToStr(uvFrontMin.u), Util.floatToStr(uvFrontMin.v)]
            textureMaxStr = [Util.floatToStr(uvFrontMax.u), Util.floatToStr(uvFrontMax.v)]

            size = bboxEntity.getScaled(entity.scale).getSizes()
            sizeXY = (size[0] + size[1]) / 2
            sizeZ = size[2]
            sizeXYStr = Util.floatToStr(sizeXY)
            sizeZStr = Util.floatToStr(sizeZ)

            center = bboxEntity.getCenter()
            centerTransformed = np.add(np.multiply(transforms3d.quaternions.rotate_vector(center, entity.rotation), entity.scale), entity.position).tolist()

            transformedBboxEntityMin = np.subtract(centerTransformed, [sizeXY / 2, sizeXY / 2, sizeZ / 2]).tolist()
            transformedBboxEntityMax = np.add(centerTransformed, [sizeXY / 2, sizeXY / 2, sizeZ / 2]).tolist()

            centerStr = [Util.floatToStr(centerTransformed[0]), Util.floatToStr(centerTransformed[1]), Util.floatToStr(centerTransformed[2])]

            bbox[diffuseSampler].extendByPoint(transformedBboxEntityMin)
            bbox[diffuseSampler].extendByPoint(transformedBboxEntityMax)

            numFrontPlanes[diffuseSampler] += 1

            if verticesFront[diffuseSampler] != "":
                verticesFront[diffuseSampler] += "\n"
            verticesFront[diffuseSampler] += "				" + centerStr[0] + " " + centerStr[1] + " " + centerStr[
                2] + normalAndColorsFront + "0.00000000 1.00000000 / " + \
                                             textureMinStr[0] + " " + textureMaxStr[1] + " / " + sizeXYStr + " " + sizeZStr + " / 1.00000000 1.00000000"
            verticesFront[diffuseSampler] += "\n				" + centerStr[0] + " " + centerStr[1] + " " + centerStr[
                2] + normalAndColorsFront + "1.00000000 1.00000000 / " + \
                                             textureMaxStr[0] + " " + textureMaxStr[1] + " / " + sizeXYStr + " " + sizeZStr + " / 1.00000000 1.00000000"
            verticesFront[diffuseSampler] += "\n				" + centerStr[0] + " " + centerStr[1] + " " + centerStr[
                2] + normalAndColorsFront + "1.00000000 0.00000000 / " + \
                                             textureMaxStr[0] + " " + textureMinStr[1] + " / " + sizeXYStr + " " + sizeZStr + " / 1.00000000 1.00000000"
            verticesFront[diffuseSampler] += "\n				" + centerStr[0] + " " + centerStr[1] + " " + centerStr[
                2] + normalAndColorsFront + "0.00000000 0.00000000 / " + \
                                             textureMinStr[0] + " " + textureMinStr[1] + " / " + sizeXYStr + " " + sizeZStr + " / 1.00000000 1.00000000"

            if slodLevel < 3 and uvTopMin is not None and uvTopMax is not None:
                numTopPlanes[diffuseSampler] += 1
                if uvMap.topZ is None:
                    topZ = centerTransformed[2]
                else:
                    topZ = (transformedBboxEntityMax[2] - transformedBboxEntityMin[2]) * (1 - uvMap.topZ) + transformedBboxEntityMin[2]
                topZStr = Util.floatToStr(topZ)
                if verticesTop[diffuseSampler] != "":
                    verticesTop[diffuseSampler] += "\n"
                verticesTop[diffuseSampler] += "				" + Util.floatToStr(transformedBboxEntityMin[0]) + " " + Util.floatToStr(
                    transformedBboxEntityMin[1]) + \
                                               " " + topZStr + normalAndColorsTop + Util.floatToStr(uvTopMin.u) + " " + Util.floatToStr(uvTopMin.v)
                verticesTop[diffuseSampler] += "\n				" + Util.floatToStr(transformedBboxEntityMax[0]) + " " + Util.floatToStr(
                    transformedBboxEntityMin[1]) + \
                                               " " + topZStr + normalAndColorsTop + Util.floatToStr(uvTopMax.u) + " " + Util.floatToStr(uvTopMin.v)
                verticesTop[diffuseSampler] += "\n				" + Util.floatToStr(transformedBboxEntityMax[0]) + " " + Util.floatToStr(
                    transformedBboxEntityMax[1]) + \
                                               " " + topZStr + normalAndColorsTop + Util.floatToStr(uvTopMax.u) + " " + Util.floatToStr(uvTopMax.v)
                verticesTop[diffuseSampler] += "\n				" + Util.floatToStr(transformedBboxEntityMin[0]) + " " + Util.floatToStr(
                    transformedBboxEntityMax[1]) + \
                                               " " + topZStr + normalAndColorsTop + Util.floatToStr(uvTopMin.u) + " " + Util.floatToStr(uvTopMax.v)

        if len(verticesFront) == 0:
            return None, None

        if slodLevel == 2 and numSkippedSlodEntities <= math.floor(0.05 * len(entities)):
            # if the slod2 model is almost the same as slod1 model then just re-use slod1 model for slod2 as well
            return nameWithoutSlodLevel + "1", None
        # but do not re-use slod1 for slod3 since slod3 models should not have xy planes
        # (so even if the entities are the same as for slod1 the slod3 model is still different)

        totalBbox = Box.createReversedInfinityBox()

        for diffuseSampler in verticesFront:
            totalBbox.extendByPoint(bbox[diffuseSampler].min)
            totalBbox.extendByPoint(bbox[diffuseSampler].max)

        self._translation = totalBbox.getCenter()

        totalBbox = totalBbox.getTranslated(np.multiply(self._translation, [-1]).tolist())

        bounds = self.createAabb(totalBbox)

        shaders = ""
        geometries = ""
        shaderIndex = 0
        for diffuseSampler in verticesFront:
            indicesTopStr = self.createIndices(numTopPlanes[diffuseSampler])
            indicesFrontStr = self.createIndices(numFrontPlanes[diffuseSampler])

            bbox[diffuseSampler] = bbox[diffuseSampler].getTranslated(np.multiply(self._translation, [-1]).tolist())
            verticesFrontStr = re.sub('(?<=\t\t\t\t)(\\S+) (\\S+) (\\S+)', self.replTranslateVertex, verticesFront[diffuseSampler])
            verticesTopStr = re.sub('(?<=\t\t\t\t)(\\S+) (\\S+) (\\S+)', self.replTranslateVertex, verticesTop[diffuseSampler])

            shaders += self.contentTemplateOdrShaderTreeLod2.replace("${DIFFUSE_SAMPLER}", diffuseSampler)

            bounds += self.createAabb(bbox[diffuseSampler])

            geometries += self.contentTemplateMeshGeometry \
                .replace("${SHADER_INDEX}", str(shaderIndex)) \
                .replace("${VERTEX_DECLARATION}", "N5A9A1E1A") \
                .replace("${INDICES.NUM}", str(numFrontPlanes[diffuseSampler] * 6)) \
                .replace("${INDICES}", indicesFrontStr) \
                .replace("${VERTICES.NUM}", str(numFrontPlanes[diffuseSampler] * 4)) \
                .replace("${VERTICES}", verticesFrontStr)
            shaderIndex += 1

            if numTopPlanes[diffuseSampler] > 0:
                shaders += self.contentTemplateOdrShaderTreeLod.replace("${DIFFUSE_SAMPLER}", diffuseSampler)

                bounds += self.createAabb(bbox[diffuseSampler])

                geometries += self.contentTemplateMeshGeometry \
                    .replace("${SHADER_INDEX}", str(shaderIndex)) \
                    .replace("${VERTEX_DECLARATION}", "NCE8F80C8") \
                    .replace("${INDICES.NUM}", str(numTopPlanes[diffuseSampler] * 6)) \
                    .replace("${INDICES}", indicesTopStr) \
                    .replace("${VERTICES.NUM}", str(numTopPlanes[diffuseSampler] * 4)) \
                    .replace("${VERTICES}", verticesTopStr)
                shaderIndex += 1

        contentModelMesh = self.contentTemplateMesh \
            .replace("${BOUNDS}\n", bounds) \
            .replace("${GEOMETRIES}\n", geometries)

        fileModelMesh = open(os.path.join(self.getOutputDirMeshes(), name.lower() + ".mesh"), 'w')
        fileModelMesh.write(contentModelMesh)
        fileModelMesh.close()

        sphere = totalBbox.getEnclosingSphere()

        contentModelOdr = self.contentTemplateOdr \
            .replace("${BBOX.MIN.X}", Util.floatToStr(totalBbox.min[0])) \
            .replace("${BBOX.MIN.Y}", Util.floatToStr(totalBbox.min[1])) \
            .replace("${BBOX.MIN.Z}", Util.floatToStr(totalBbox.min[2])) \
            .replace("${BBOX.MAX.X}", Util.floatToStr(totalBbox.max[0])) \
            .replace("${BBOX.MAX.Y}", Util.floatToStr(totalBbox.max[1])) \
            .replace("${BBOX.MAX.Z}", Util.floatToStr(totalBbox.max[2])) \
            .replace("${BSPHERE.CENTER.X}", Util.floatToStr(sphere.center[0])) \
            .replace("${BSPHERE.CENTER.Y}", Util.floatToStr(sphere.center[1])) \
            .replace("${BSPHERE.CENTER.Z}", Util.floatToStr(sphere.center[2])) \
            .replace("${BSPHERE.RADIUS}", Util.floatToStr(sphere.radius)) \
            .replace("${NAME}", name.lower()) \
            .replace("${SHADERS}\n", shaders)

        fileModelOdr = open(os.path.join(self.getOutputDirModels(), name.lower() + ".odr"), 'w')
        fileModelOdr.write(contentModelOdr)
        fileModelOdr.close()

        if slodLevel == 1:
            # TODO get cluster index and use this:
            # itemHdDistance = getLodDistanceForSLodModel(i)
            itemHdDistance = LodMapCreator.LOD_DISTANCE
            itemLodDistance = LodMapCreator.SLOD_DISTANCE
        elif slodLevel == 2:
            itemHdDistance = LodMapCreator.SLOD_DISTANCE
            itemLodDistance = LodMapCreator.SLOD2_DISTANCE
        elif slodLevel == 3:
            itemHdDistance = LodMapCreator.SLOD2_DISTANCE
            itemLodDistance = LodMapCreator.SLOD3_DISTANCE
        else:
            Exception("unknown slod level " + str(slodLevel))

        if not os.path.exists(os.path.join(self.getOutputDirModels(), self.getYtypName() + ".ytyp.xml")):
            self.slodYtypItems = open(os.path.join(self.getOutputDirModels(), self.getYtypName() + ".ytyp.xml"), 'w')
            self.slodYtypItems.write("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<CMapTypes>
  <extensions/>
  <archetypes>
""")
        self.slodYtypItems.write(self.replacePlaceholders(self.contentTemplateYtypItem, name, totalBbox, itemHdDistance, itemLodDistance))

        return name, self._translation

    def fixHdOrOrphanHdLodLevelsAndRearrangeEntites(self, content: str) -> str:
        hdEntities = ""
        orphanHdEntites = ""
        for match in re.finditer('(\\s*<Item type="CEntityDef">' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*<flags value=")([^"]+)("\\s*/>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*<parentIndex value="([^"]+)"\\s*/>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*<lodLevel>)[^<]+(</lodLevel>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*<priorityLevel>)([^<]+)(</priorityLevel>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*</Item>)', content):

            isOrphanHd = (match.group(4) == "-1")

            flags = int(match.group(2))
            if isOrphanHd:
                # TODO pruefen ob korrekt
                flags |= Flag.FLAGS_ORPHANHD_DEFAULT
                flags &= ~Flag.FLAGS_ORPHANHD_EXCLUDE_DEFAULT
            else:
                flags |= Flag.FLAGS_HD_DEFAULT
                flags &= ~Flag.FLAGS_HD_EXCLUDE_DEFAULT

            entity = match.group(1) + str(flags) + match.group(3) + (LodLevel.ORPHAN_HD if isOrphanHd else LodLevel.HD) + match.group(5) + \
                     (match.group(6) if isOrphanHd else PriorityLevel.REQUIRED) + match.group(7)

            if isOrphanHd:
                orphanHdEntites += entity
            else:
                hdEntities += entity

        start = re.search("<entities>", content).end()
        end = re.search("\\s+</entities>[\\S\\s]*?\\Z", content, re.M).start()

        return content[:start] + hdEntities + orphanHdEntites + content[end:]

    def resetParentIndexAndNumChildren(self, content: str) -> str:
        result = re.sub('(<parentIndex value=")[^"]+("/>)', '\\g<1>-1\\g<2>', content)
        result = re.sub('(<numChildren value=")[^"]+("/>)', '\\g<1>0\\g<2>', result)
        return result

    def getLodDistanceForSlodModel(self, slodCluster: int) -> float:
        slodFurthestDistance = self._slodEntitiesFurthestDistances[slodCluster]
        # it should never happen that LOD_DISTANCE + slodFurthestDistance
        # exceeds SLOD_DISTANCE but to be absolutely sure there is that min()
        return min(LodMapCreator.SLOD_DISTANCE - 1, round(LodMapCreator.LOD_DISTANCE + slodFurthestDistance))

    def determinePrefixBundles(self):
        self.bundlePrefixes = []
        numMatches = 0
        candidate = None
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml") or filename.endswith("_lod.ymap.xml"):
                continue

            mapName = filename[:-9]

            if candidate is None:
                candidate = mapName

            parts = [candidate.rstrip("_"), "", ""]
            while not mapName.startswith(parts[0]):
                parts = parts[0].rpartition("_")

            newCandidate = parts[0]

            if newCandidate == "":
                if numMatches > 0:
                    self.bundlePrefixes.append(candidate)
                numMatches = 1
                candidate = mapName
            elif newCandidate == candidate:
                numMatches += 1
            else:
                if numMatches > 1:
                    self.bundlePrefixes.append(candidate)
                    numMatches = 1
                    candidate = mapName
                else:
                    numMatches += 1
                    candidate = newCandidate

        if candidate != "" and numMatches > 0:
            self.bundlePrefixes.append(candidate)

    def processFiles(self):
        for mapPrefix in self.bundlePrefixes:
            self.processFilesWithPrefix(mapPrefix)

        if self.slodYtypItems is not None:
            self.slodYtypItems.write("""  </archetypes>
      <name>""" + self.getYtypName() + """</name>
      <dependencies/>
      <compositeEntityTypes/>
    </CMapTypes>""")

            self.slodYtypItems.close()

    def processFilesWithPrefix(self, mapPrefix: str):
        slod2Index = 0
        slod3Index = 0
        contentEntitiesSlod2 = ""
        contentEntitiesSlod3 = ""
        totalNumClusters = 0
        mapNameSlod2 = mapPrefix.lower() + "_slod2"

        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml") or not filename.startswith(mapPrefix.lower()):
                continue

            mapName = filename[:-9]
            mapNameLod = mapName + "_lod"

            filenameLod = mapNameLod + ".ymap.xml"

            if os.path.exists(os.path.join(self.getOutputDirMaps(), mapName + ".ymap.xml")):
                print("\twarning: skipping " + filename + " since such a map was created by this script")
                continue

            print("\tprocessing " + filename)

            fileNoLod = open(os.path.join(self.inputDir, filename), 'r')
            contentNoLod = fileNoLod.read()
            fileNoLod.close()

            contentNoLod = self.resetParentIndexAndNumChildren(contentNoLod)

            contentNoLod = Ymap.calculateAndReplaceLodDistanceForEntitiesWithLod(contentNoLod, self.ytypItems)

            pattern = re.compile('[\t ]*<Item type="CEntityDef">' +
                                 '\\s*<archetypeName>([^<]+)</archetypeName>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*<lodDist value="([^"]+)"\\s*/>' +
                                 '(?:\\s*<[^/].*>)*' +
                                 '\\s*</Item>[\r\n]+')

            lodCoords = []
            slodCoords = []
            for matchobj in re.finditer(pattern, contentNoLod):
                archetypeName = matchobj.group(1)
                if self.getLod(archetypeName) is None:
                    continue

                lodCoords.append([float(matchobj.group(2)), float(matchobj.group(3)), float(matchobj.group(4))])

                lodDistance = float(matchobj.group(5))

                if archetypeName in self.slodCandidates and lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD1:
                    slodCoords.append([float(matchobj.group(2)), float(matchobj.group(3)), float(matchobj.group(4))])

            foundLodModel = len(lodCoords) > 0
            if not foundLodModel:
                contentNoLod = self.fixHdOrOrphanHdLodLevelsAndRearrangeEntites(contentNoLod)

                fileNoLod = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
                fileNoLod.write(contentNoLod)
                fileNoLod.close()
                continue

            self._foundSlodModel = len(slodCoords) > 0
            if self._foundSlodModel:
                self._clusters, self._slodEntitiesFurthestDistances = \
                    Util.performClustering(slodCoords, LodMapCreator.NUM_CHILDREN_MAX_VALUE, LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL)

                numClusters = len(np.unique(self._clusters))

                self._entitiesForSlod = []
                for cluster in range(numClusters):
                    self._entitiesForSlod.append([])

                self._slodIndex = 0
                totalNumClusters += numClusters

            contentLod = re.sub('(\\s*<Item type="CEntityDef">' +
                                '\\s*<archetypeName>)([^<]+)(</archetypeName>' +
                                '\\s*<flags value=")([^"]+)("\\s*/>' +
                                '(?:\\s*<[^/].*>)*' +
                                '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
                                '\\s*<rotation x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"\\s*/>' +
                                '\\s*<scaleXY value="([^"]+)"\\s*/>' +
                                '\\s*<scaleZ value="([^"]+)"\\s*/>' +
                                '\\s*<parentIndex)(?:\\s*/>| value="[^"]+")(\\s*/>' +
                                '\\s*<lodDist value=")([^"]+)("\\s*/>' +
                                '\\s*<childLodDist value=")[^"]+("\\s*/>' +
                                '\\s*<lodLevel>)[^<]+(</lodLevel>' +
                                '\\s*<numChildren value=")[^"]+("\\s*/>' +
                                '\\s*<priorityLevel>)[^<]+(</priorityLevel>' +
                                '(?:\\s*<[^/].*>)*' +
                                '\\s*</Item>)', self.replLod, contentNoLod, flags=re.M)

            contentLod = self.replaceName(contentLod, mapNameLod)
            contentLod = self.replaceParent(contentLod, mapNameSlod2 if self._foundSlodModel else None)
            contentFlag = ContentFlag.LOD
            if self._foundSlodModel:
                contentFlag |= ContentFlag.SLOD
            contentLod = self.replaceFlagsAndContentFlags(contentLod, ContentFlag.LOD, contentFlag)

            if self._foundSlodModel:
                contentEntitiesSlod = ""
                for i in range(len(self._entitiesForSlod)):
                    slodNameNumbering = "_" + str(i)

                    clusterEntries = np.where(self._clusters == i)
                    clusterSize = len(clusterEntries[0])

                    lodDistance = self.getLodDistanceForSlodModel(i)

                    slodName = mapPrefix + mapName[len(mapPrefix):] + slodNameNumbering + "_SLOD"
                    archetypeNameSlod1, positionSlod1 = self.createSlodModel(slodName, 1, self._entitiesForSlod[i])
                    archetypeNameSlod2, positionSlod2 = self.createSlodModel(slodName, 2, self._entitiesForSlod[i])

                    contentEntitiesSlod += self.contentTemplateEntitySlod \
                        .replace("${POSITION.X}", Util.floatToStr(positionSlod1[0])) \
                        .replace("${POSITION.Y}", Util.floatToStr(positionSlod1[1])) \
                        .replace("${POSITION.Z}", Util.floatToStr(positionSlod1[2])) \
                        .replace("${NAME}", archetypeNameSlod1) \
                        .replace("${NUM_CHILDREN}", str(clusterSize)) \
                        .replace("${PARENT_INDEX}", "-1" if archetypeNameSlod2 is None else str(slod2Index)) \
                        .replace("${LOD_LEVEL}", LodLevel.SLOD1) \
                        .replace("${CHILD.LOD_DISTANCE}", Util.floatToStr(lodDistance)) \
                        .replace("${LOD_DISTANCE}", Util.floatToStr(LodMapCreator.SLOD_DISTANCE)) \
                        .replace("${FLAGS}", str(Flag.FLAGS_SLOD1))

                    if archetypeNameSlod2 is not None:
                        if positionSlod2 is None:
                            positionSlod2 = positionSlod1

                        archetypeNameSlod3, positionSlod3 = self.createSlodModel(slodName, 3, self._entitiesForSlod[i])

                        contentEntitiesSlod2 += self.contentTemplateEntitySlod \
                            .replace("${POSITION.X}", Util.floatToStr(positionSlod2[0])) \
                            .replace("${POSITION.Y}", Util.floatToStr(positionSlod2[1])) \
                            .replace("${POSITION.Z}", Util.floatToStr(positionSlod2[2])) \
                            .replace("${NAME}", archetypeNameSlod2) \
                            .replace("${NUM_CHILDREN}", "1") \
                            .replace("${PARENT_INDEX}", "-1" if archetypeNameSlod3 is None else str(slod3Index)) \
                            .replace("${LOD_LEVEL}", LodLevel.SLOD2) \
                            .replace("${CHILD.LOD_DISTANCE}", Util.floatToStr(LodMapCreator.SLOD_DISTANCE)) \
                            .replace("${LOD_DISTANCE}", Util.floatToStr(LodMapCreator.SLOD2_DISTANCE)) \
                            .replace("${FLAGS}", str(Flag.FLAGS_SLOD2))

                        slod2Index += 1

                        if archetypeNameSlod3 is not None:
                            contentEntitiesSlod3 += self.contentTemplateEntitySlod \
                                .replace("${POSITION.X}", Util.floatToStr(positionSlod3[0])) \
                                .replace("${POSITION.Y}", Util.floatToStr(positionSlod3[1])) \
                                .replace("${POSITION.Z}", Util.floatToStr(positionSlod3[2])) \
                                .replace("${NAME}", archetypeNameSlod3) \
                                .replace("${NUM_CHILDREN}", "1") \
                                .replace("${PARENT_INDEX}", "-1") \
                                .replace("${LOD_LEVEL}", LodLevel.SLOD3) \
                                .replace("${CHILD.LOD_DISTANCE}", Util.floatToStr(LodMapCreator.SLOD2_DISTANCE)) \
                                .replace("${LOD_DISTANCE}", Util.floatToStr(LodMapCreator.SLOD3_DISTANCE)) \
                                .replace("${FLAGS}", str(Flag.FLAGS_SLOD3))

                            slod3Index += 1

                contentLod = contentLod.replace("<entities>\n", "<entities>\n" + contentEntitiesSlod)

            f = open(os.path.join(self.getOutputDirMaps(), filenameLod), 'w')
            f.write(contentLod)
            f.close()

            contentNoLod = self.replaceParent(contentNoLod, mapNameLod)

            # <!--
            # fix parent name and parentIndex in hd map to match lod map
            self._index = len(self._entitiesForSlod) if self._foundSlodModel else 0

            contentNoLod = re.sub('(\\s*<Item type="CEntityDef">' +
                                  '\\s*<archetypeName>([^<]+)</archetypeName>' +
                                  '(?:\\s*<[^/].*>)*' +
                                  '\\s*<parentIndex value=")[^"]+("\\s*/>' +
                                  '(?:\\s*<[^/].*>)*' +
                                  '\\s*</Item>)', self.replParentIndex, contentNoLod, flags=re.M)
            # -->

            contentNoLod = self.fixHdOrOrphanHdLodLevelsAndRearrangeEntites(contentNoLod)

            fileNoLod = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
            fileNoLod.write(contentNoLod)
            fileNoLod.close()

        if contentEntitiesSlod2 or contentEntitiesSlod3:
            # <!--
            # fix parentIndex in lod map to match slod map
            for filename in os.listdir(self.getOutputDirMaps()):
                if not filename.endswith("_lod.ymap.xml") or not filename.startswith(mapPrefix.lower()):
                    continue

                file = open(os.path.join(self.getOutputDirMaps(), filename), 'r')
                content_new = file.read()
                file.close()

                content_new = re.sub('(\\s*<Item type="CEntityDef">' +
                                     '\\s*<archetypeName>[^<]+_(?i:slod)1</archetypeName>' +
                                     '(?:\\s*<[^/].*>)*' +
                                     '\\s*<parentIndex value=")([^"]+)("\\s*/>' +
                                     '(?:\\s*<[^/].*>)*' +
                                     '\\s*</Item>)',
                    lambda match: match.group(0) if int(match.group(2)) < 0 else match.group(1) + str(int(match.group(2)) + slod3Index) + match.group(3),
                    content_new, flags=re.M)

                file = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
                file.write(content_new)
                file.close()
            # -->

            slod2Map = self.contentTemplateSlod2Map \
                .replace("${NAME}", mapNameSlod2) \
                .replace("${ENTITIES}\n", contentEntitiesSlod3 + contentEntitiesSlod2)

            fileSlod2Map = open(os.path.join(self.getOutputDirMaps(), mapNameSlod2 + ".ymap.xml"), 'w')
            fileSlod2Map.write(slod2Map)
            fileSlod2Map.close()

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.getOutputDirMaps(),
            lambda filename: not filename.endswith(".ymap.xml") or not filename.startswith(tuple(each.lower() for each in self.bundlePrefixes)))

    # adapt extents and set current datetime
    def fixMapExtents(self):
        print("\tfixing map extents")

        if self.slodYtypItems is not None:
            self.ytypItems |= YtypParser.readYtypDirectory(self.getOutputDirModels())

        for filename in os.listdir(self.getOutputDirMaps()):
            if not filename.endswith(".ymap.xml"):
                continue

            file = open(os.path.join(self.getOutputDirMaps(), filename), 'r')
            content = file.read()
            file.close()

            content = Ymap.fixMapExtents(content, self.ytypItems)

            file = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
            file.write(content)
            file.close()
