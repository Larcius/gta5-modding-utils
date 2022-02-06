import shutil
from typing import IO, Optional
from natsort import natsorted
import os
import math
from dataclasses import dataclass

from common.Box import Box
from common.Util import Util
from common.texture.UV import UV
from common.ytyp.YtypItem import YtypItem
from common.ytyp.YtypParser import YtypParser


class LodModelCreator:
    inputDir: str
    outputDir: str

    prefix: str

    template2PlanesLodOdr: str
    template2PlanesLodMesh: str
    template3PlanesLodOdr: str
    template3PlanesLodMesh: str
    templateLodYtypItem: str

    ytyps: dict[str, YtypItem]
    lodYtypItems: Optional[IO]

    # needed to avoid getting wrong pixels from texture (e.g. using 0 would result in getting right-most resp. down-most pixel)
    TEXTURE_UV_EPS = 1 / 256

    @dataclass
    class Tree:
        name: str
        texture_origin: float = 0.5
        planeZ: Optional[float] = 0.5
        uvFrontMin: UV = UV(0, 0)
        uvFrontMax: UV = UV(1, 1)
        uvTopMin: Optional[UV] = None
        uvTopMax: Optional[UV] = None
        uvTopCenter: Optional[UV] = None
        uvTopCenterZ: Optional[float] = None
        uvSideMin: Optional[UV] = None
        uvSideMax: Optional[UV] = None
        _textureOriginSide: Optional[float] = None

        def textureOriginSide(self) -> Optional[float]:
            return self.texture_origin if self._textureOriginSide is None else self._textureOriginSide

        def texture(self) -> str:
            return "lod_" + self.name.lower()

        def hasDedicatedSideTexture(self) -> bool:
            return self.uvSideMin is not None and self.uvSideMax is not None

        def getTextureDictionary(self, prefix: str) -> str:
            if self.name.startswith("prop_bush_"):
                dictMain = "bushes"
            elif self.name.startswith("prop_palm_") or self.name.startswith("prop_fan_palm_"):
                dictMain = "palms"
            else:
                dictMain = "trees"
            return prefix + "_" + dictMain + "_lod"

    trees: dict[str, Tree]

    def __init__(self, inputDir: str, outputDir: str, prefix: str):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix

    def run(self):
        print("running lod model creator...")
        self.readTemplates()
        self.readYtyps()
        self.prepareLodCandidates()
        self.createOutputDir()
        self.processFiles()
        self.copyTextureDictionaries()
        print("lod model creator DONE")

    def getOutputModelsDir(self):
        return os.path.join(self.outputDir, "models")

    def getOutputMeshesDir(self):
        return os.path.join(self.outputDir, "_lod_meshes")

    def getYtypName(self) -> str:
        return self.prefix + "_lod"

    def prepareLodCandidates(self):
        trees = {
            # trees
            "prop_tree_birch_01": LodModelCreator.Tree("prop_tree_birch_01", 0.546875, 0.6640625, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1),
                UV(0.7734375, 0.5078125)),
            "prop_tree_birch_02": LodModelCreator.Tree("prop_tree_birch_02", 0.421875, 0.5703125, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1),
                UV(0.765625, 0.3671875)),
            "prop_tree_birch_03": LodModelCreator.Tree("prop_tree_birch_03", 0.546875),
            "prop_tree_birch_03b": LodModelCreator.Tree("prop_tree_birch_03b", 0.5625),
            "prop_tree_birch_04": LodModelCreator.Tree("prop_tree_birch_04", 0.5625, 0.421875, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0),
                UV(0.7734375, 0.453125)),
            "prop_tree_maple_02": LodModelCreator.Tree("prop_tree_maple_02", 0.421875),
            # TODO provide top view
            "prop_tree_maple_03": LodModelCreator.Tree("prop_tree_maple_03", 0.5),
            "prop_tree_cedar_02": LodModelCreator.Tree("prop_tree_cedar_02", 0.515625, 0.46315789473, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_cedar_03": LodModelCreator.Tree("prop_tree_cedar_03", 0.5390625, 0.51052631578, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_cedar_04": LodModelCreator.Tree("prop_tree_cedar_04", 0.484375, 0.3947368421, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1),
                UV(0.484375, 0.87890625)),
            "prop_tree_cedar_s_01": LodModelCreator.Tree("prop_tree_cedar_s_01", 0.484375, 0.71875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1),
                UV(0.46875, 0.8203125)),
            "prop_tree_cedar_s_04": LodModelCreator.Tree("prop_tree_cedar_s_04", 0.5, 0.68269230769, UV(0, 0), UV(1, 0.8125), UV(0, 1), UV(1, 0.8125)),
            "prop_tree_cedar_s_05": LodModelCreator.Tree("prop_tree_cedar_s_05", 0.46875),
            "prop_tree_cedar_s_06": LodModelCreator.Tree("prop_tree_cedar_s_06", 0.5),
            "prop_tree_cypress_01": LodModelCreator.Tree("prop_tree_cypress_01", 0.5),
            "prop_tree_eng_oak_01": LodModelCreator.Tree("prop_tree_eng_oak_01", 0.5, 0.5703125, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.75, 0.53125)),
            "prop_tree_eucalip_01": LodModelCreator.Tree("prop_tree_eucalip_01", 0.5, 0.359375, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1)),
            "prop_tree_jacada_01": LodModelCreator.Tree("prop_tree_jacada_01", 0.484375, 0.46875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_jacada_02": LodModelCreator.Tree("prop_tree_jacada_02", 0.515625, 0.546875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_oak_01": LodModelCreator.Tree("prop_tree_oak_01", 0.4765625, 0.4765625, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1), UV(0.46875, 0.75)),
            "prop_tree_olive_01": LodModelCreator.Tree("prop_tree_olive_01", 0.5, 0.40625, UV(0, 0), UV(1, 0.5), UV(0, 1), UV(1, 0.5)),
            "prop_tree_pine_01": LodModelCreator.Tree("prop_tree_pine_01", 0.515625, 0.515625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.515625, 0.79296875)),
            "prop_tree_pine_02": LodModelCreator.Tree("prop_tree_pine_02", 0.546875, 0.6875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.5, 0.80078125)),
            "prop_tree_fallen_pine_01": LodModelCreator.Tree("prop_tree_fallen_pine_01", 0.609375, 1, UV(0, 0), UV(1, 0.625)),
            # , UV(0, 1), UV(1, 0.625), UV(0.390625, 0.7734375)),
            "prop_s_pine_dead_01": LodModelCreator.Tree("prop_s_pine_dead_01", 0.40625, 0.4875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.53125, 0.8515625)),
            "prop_w_r_cedar_01": LodModelCreator.Tree("prop_w_r_cedar_01", 0.5, 0.8, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_w_r_cedar_dead": LodModelCreator.Tree("prop_w_r_cedar_dead", 0.59375, 0.425, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.53125, 0.78125)),
            "test_tree_cedar_trunk_001": LodModelCreator.Tree("test_tree_cedar_trunk_001", 0.515625, 0.5769231, UV(0, 0), UV(1, 0.8125), UV(0, 1),
                UV(1, 0.8125)),
            "test_tree_forest_trunk_01": LodModelCreator.Tree("test_tree_forest_trunk_01", 0.515625, 0.3894231, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125),
                UV(1, 1)),
            "test_tree_forest_trunk_04": LodModelCreator.Tree("test_tree_forest_trunk_04", 0.453125, 1, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0),
                UV(0.78125, 0.5546875), 0.6484375),
            # trees2
            "prop_tree_lficus_02": LodModelCreator.Tree("prop_tree_lficus_02", 0.4453125, 0.55, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_03": LodModelCreator.Tree("prop_tree_lficus_03", 0.46875, 0.359375, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_05": LodModelCreator.Tree("prop_tree_lficus_05", 0.46875, 0.3125, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_06": LodModelCreator.Tree("prop_tree_lficus_06", 0.453125, 0.43, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1), None, 0.15),
            "prop_tree_mquite_01": LodModelCreator.Tree("prop_tree_mquite_01", 0.46875),
            "prop_rio_del_01": LodModelCreator.Tree("prop_rio_del_01", 0.53125),
            "prop_rus_olive": LodModelCreator.Tree("prop_rus_olive", 0.484375, 0.53125, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            # bushes
            "prop_bush_lrg_04b": LodModelCreator.Tree("prop_bush_lrg_04b", 0.37333333, 0.40625, UV(0, 0), UV(0.5859375, 0.5), UV(0, 0.5), UV(1, 1), None, None,
                UV(0.5859375, 0), UV(1, 0.5), 0.45283018867),
            "prop_bush_lrg_04c": LodModelCreator.Tree("prop_bush_lrg_04c", 0.37333333, 0.4375, UV(0, 0), UV(0.5859375, 0.5), UV(0, 0.5), UV(1, 1), None, None,
                UV(0.5859375, 0), UV(1, 0.5), 0.5660377358),
            "prop_bush_lrg_04d": LodModelCreator.Tree("prop_bush_lrg_04d", 0.4, 0.421875, UV(0, 0), UV(0.5078125, 0.5), UV(0, 0.5), UV(0.75, 1), None, None,
                UV(0.5078125, 0), UV(1, 0.484375), 0.47619047619),
            # palms
            "prop_palm_fan_02_b": LodModelCreator.Tree("prop_palm_fan_02_b", 0.515625, 0.25625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.484375, 0.8125)),
            "prop_palm_fan_03_c": LodModelCreator.Tree("prop_palm_fan_03_c", 0.5, 0.166666667, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_palm_fan_03_d": LodModelCreator.Tree("prop_palm_fan_03_d", 0.484375, 0.15865384615, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1),
                UV(0.484375, 0.90625)),
            "prop_palm_fan_04_b": LodModelCreator.Tree("prop_palm_fan_04_b", 0.484375, 0.321875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1),
                UV(0.46875, 0.80859375)),
            "prop_palm_huge_01a": LodModelCreator.Tree("prop_palm_huge_01a", 0.46875, 0.11574074074, UV(0, 0), UV(1, 0.84375), UV(0, 1), UV(1, 0.84375),
                UV(0.484375, 0.921875)),
            "prop_palm_med_01b": LodModelCreator.Tree("prop_palm_med_01b", 0.515625, 0.24431818181, UV(0, 0), UV(1, 0.6875), UV(0, 1), UV(1, 0.6875),
                UV(0.546875, 0.84375)),
            "prop_palm_sm_01e": LodModelCreator.Tree("prop_palm_sm_01e", 0.515625, None, UV(0, 0), UV(1, 1)),
        }
        # add other Props that should use the same UV mapping
        trees["prop_palm_sm_01d"] = trees["prop_palm_sm_01f"] = \
            trees["prop_palm_sm_01e"]
        trees["prop_palm_med_01a"] = trees["prop_palm_med_01c"] = \
            trees["prop_palm_med_01b"]
        trees["prop_fan_palm_01a"] = trees["prop_palm_fan_02_a"] = \
            trees["prop_palm_fan_02_b"]
        trees["prop_palm_sm_01a"] = trees["prop_palm_fan_04_a"] = trees["prop_palm_fan_04_b"] = \
            trees["prop_palm_fan_04_b"]
        trees["prop_palm_fan_03_a"] = trees["prop_palm_fan_03_b"] = trees["prop_palm_fan_03_c_graff"] = trees["prop_palm_fan_04_c"] = \
            trees["prop_palm_fan_03_c"]
        trees["prop_palm_med_01d"] = trees["prop_palm_fan_03_d_graff"] = trees["prop_palm_fan_04_d"] = \
            trees["prop_palm_fan_03_d"]
        trees["prop_palm_huge_01b"] = \
            trees["prop_palm_huge_01a"]

        self.trees = trees

    def readTemplates(self):
        self.template2PlanesLodOdr = Util.readFile(os.path.join(os.path.dirname(__file__), 'templates', 'template_2_planes_lod.odr'))
        self.template2PlanesLodMesh = Util.readFile(os.path.join(os.path.dirname(__file__), 'templates', 'template_2_planes_lod.mesh'))
        self.template3PlanesLodOdr = Util.readFile(os.path.join(os.path.dirname(__file__), 'templates', 'template_3_planes_lod.odr'))
        self.template3PlanesLodMesh = Util.readFile(os.path.join(os.path.dirname(__file__), 'templates', 'template_3_planes_lod.mesh'))
        self.templateLodYtypItem = Util.readFile(os.path.join(os.path.dirname(__file__), 'templates', 'template_lod_ytyp_item.xml'))

    def readYtyps(self):
        self.ytyps = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "../..", "resources", "ytyp"))

    def createOutputDir(self):
        if os.path.exists(self.outputDir):
            raise ValueError("Output dir " + self.outputDir + " must not exist")

        os.makedirs(self.outputDir)
        os.mkdir(self.getOutputModelsDir())
        os.mkdir(self.getOutputMeshesDir())

    @staticmethod
    def createTextureUvWithEps(minUv: UV, maxUv: UV) -> (Optional[float], Optional[float]):
        if minUv is None or maxUv is None:
            return minUv, maxUv

        minUvEps = UV(minUv.u, minUv.v)
        maxUvEps = UV(maxUv.u, maxUv.v)
        if minUvEps.u < maxUvEps.u:
            minUvEps.u += LodModelCreator.TEXTURE_UV_EPS
        else:
            maxUvEps.u += LodModelCreator.TEXTURE_UV_EPS

        if minUvEps.v < maxUvEps.v:
            minUvEps.v += LodModelCreator.TEXTURE_UV_EPS
        else:
            maxUvEps.v += LodModelCreator.TEXTURE_UV_EPS

        return minUvEps, maxUvEps

    def replacePlaceholders(self, template: str, propName: str, tree: Tree, bbox: Box, planeIntersection: list[float]) -> str:
        bsphere = bbox.getEnclosingSphere()
        bboxSize = bbox.getSizes()

        uvFrontMin, uvFrontMax = self.createTextureUvWithEps(tree.uvFrontMin, tree.uvFrontMax)
        uvTopMin, uvTopMax = self.createTextureUvWithEps(tree.uvTopMin, tree.uvTopMax)
        uvSideMin, uvSideMax = self.createTextureUvWithEps(tree.uvSideMin, tree.uvSideMax)
        if uvSideMin is None:
            uvSideMin = uvFrontMin
        if uvSideMax is None:
            uvSideMax = uvFrontMax

        result = template \
            .replace("${TEXTURE_DICTIONARY}", tree.getTextureDictionary(self.prefix)) \
            .replace("${MESH_FILENAME}", os.path.join(os.path.relpath(self.getOutputMeshesDir(), self.getOutputModelsDir()), propName.lower() + "_lod.mesh")) \
            .replace("${PROP_NAME.NORMAL}", propName) \
            .replace("${BSPHERE.CENTER.X}", str(bsphere.center[0])) \
            .replace("${BSPHERE.CENTER.Y}", str(bsphere.center[1])) \
            .replace("${BSPHERE.CENTER.Z}", str(bsphere.center[2])) \
            .replace("${BSPHERE.RADIUS}", str(bsphere.radius)) \
            .replace("${BBOX.SIZE.MAX_XY}", str(max(bboxSize[0], bboxSize[1]))) \
            .replace("${BBOX.SIZE.X}", str(bboxSize[0])) \
            .replace("${BBOX.SIZE.Y}", str(bboxSize[1])) \
            .replace("${BBOX.SIZE.Z}", str(bboxSize[2])) \
            .replace("${BBOX.MIN.X}", str(bbox.min[0])) \
            .replace("${BBOX.MIN.Y}", str(bbox.min[1])) \
            .replace("${BBOX.MIN.Z}", str(bbox.min[2])) \
            .replace("${BBOX.MAX.X}", str(bbox.max[0])) \
            .replace("${BBOX.MAX.Y}", str(bbox.max[1])) \
            .replace("${BBOX.MAX.Z}", str(bbox.max[2])) \
            .replace("${PLANE_INTERSECTION.X}", str(planeIntersection[0])) \
            .replace("${PLANE_INTERSECTION.Y}", str(planeIntersection[1])) \
            .replace("${BBOX.SIZE.X / 2}", str(bboxSize[0] / 2)) \
            .replace("${BBOX.SIZE.Y / 2}", str(bboxSize[1] / 2)) \
            .replace("${BBOX.SIZE.Z / 2}", str(bboxSize[2] / 2)) \
            .replace("${math.sqrt(BBOX.SIZE.X**2 + BBOX.SIZE.Z**2) / 2}", str(math.sqrt(bboxSize[0] ** 2 + bboxSize[2] ** 2) / 2)) \
            .replace("${TEXTURE.FRONT.MIN.U}", str(uvFrontMin.u)) \
            .replace("${TEXTURE.FRONT.MIN.V}", str(uvFrontMin.v)) \
            .replace("${TEXTURE.FRONT.MAX.U}", str(uvFrontMax.u)) \
            .replace("${TEXTURE.FRONT.MAX.V}", str(uvFrontMax.v)) \
            .replace("${TEXTURE.SIDE.MIN.U}", str(uvSideMin.u)) \
            .replace("${TEXTURE.SIDE.MIN.V}", str(uvSideMin.v)) \
            .replace("${TEXTURE.SIDE.MAX.U}", str(uvSideMax.u)) \
            .replace("${TEXTURE.SIDE.MAX.V}", str(uvSideMax.v)) \
            .replace("${TEXTURE.NAME}", tree.texture())

        if uvTopMin is not None and uvTopMax is not None:
            planeMinZ = bbox.min[2] + (bbox.max[2] - bbox.min[2]) * (1 - tree.planeZ)
            if tree.uvTopCenterZ is None:
                planeMaxZ = min(bbox.max[2] - min(bboxSize) * 0.1, planeMinZ + bboxSize[0] / 2)
            else:
                planeMaxZ = bbox.min[2] + bboxSize[2] * (1 - tree.uvTopCenterZ)

            if tree.uvTopCenter is None:
                minUv = UV(min(uvTopMin.u, uvTopMax.u), min(uvTopMin.v, uvTopMax.v))
                maxUv = UV(max(uvTopMin.u, uvTopMax.u), max(uvTopMin.v, uvTopMax.v))
                if uvTopMin.u > tree.uvTopMax.u:
                    topCenter = UV(minUv.v + (maxUv.v - minUv.v) * (1 - tree.texture_origin), minUv.u + (maxUv.u - minUv.u) * tree.textureOriginSide())
                else:
                    topCenter = UV(minUv.u + (maxUv.u - minUv.u) * tree.texture_origin, minUv.v + (maxUv.v - minUv.v) * (1 - tree.textureOriginSide()))
            else:
                topCenter = tree.uvTopCenter

            result = result \
                .replace("${PLANE.MIN.Z}", str(planeMinZ)) \
                .replace("${PLANE.MAX.Z}", str(planeMaxZ)) \
                .replace("${TEXTURE.TOP.CENTER.U}", str(topCenter.u)) \
                .replace("${TEXTURE.TOP.CENTER.V}", str(topCenter.v)) \
                .replace("${TEXTURE.TOP.MIN.U}", str(uvTopMin.u)) \
                .replace("${TEXTURE.TOP.MIN.V}", str(uvTopMin.v)) \
                .replace("${TEXTURE.TOP.MAX.U}", str(uvTopMax.u)) \
                .replace("${TEXTURE.TOP.MAX.V}", str(uvTopMax.v))

        return result

    def createLodModel(self, prop: str, tree: Tree, bbox: Box):
        sizes = bbox.getSizes()

        sizeX = sizes[0]
        sizeY = sizes[1]
        planeIntersection = [bbox.min[0] + sizeX * tree.texture_origin, bbox.min[1] + sizeY * tree.textureOriginSide()]

        if tree.uvTopMin is None or tree.uvTopMax is None:
            templateOdr = self.template2PlanesLodOdr
            templateMesh = self.template2PlanesLodMesh
        else:
            templateOdr = self.template3PlanesLodOdr
            templateMesh = self.template3PlanesLodMesh

        open(os.path.join(self.getOutputModelsDir(), prop.lower() + "_lod.odr"), 'w') \
            .write(self.replacePlaceholders(templateOdr, prop, tree, bbox, planeIntersection))

        open(os.path.join(self.getOutputMeshesDir(), prop.lower() + "_lod.mesh"), 'w') \
            .write(self.replacePlaceholders(templateMesh, prop, tree, bbox, planeIntersection))

        self.lodYtypItems.write(self.replacePlaceholders(self.templateLodYtypItem, prop, tree, bbox, planeIntersection))

    def processFiles(self):
        self.lodYtypItems = open(os.path.join(self.getOutputModelsDir(), self.getYtypName() + ".ytyp.xml"), 'w')

        self.lodYtypItems.write("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<CMapTypes>
  <extensions/>
  <archetypes>
""")

        for name in natsorted(self.trees):
            if name not in self.ytyps:
                print("WARNING: could not find archetype " + name + " in any of the provided ytyp files. Skipping lod creation for this archetype.")
                continue

            tree = self.trees[name]
            ytypItem = self.ytyps[name]

            self.createLodModel(name, tree, ytypItem.boundingBox)

        self.lodYtypItems.write("""  </archetypes>
  <name>""" + self.getYtypName() + """</name>
  <dependencies/>
  <compositeEntityTypes/>
</CMapTypes>""")

        self.lodYtypItems.close()

    def copyTextureDictionaries(self):
        texturesDir = os.path.join(os.path.dirname(__file__), "textures")
        for filename in os.listdir(texturesDir):
            shutil.copyfile(os.path.join(texturesDir, filename), os.path.join(self.getOutputModelsDir(), self.prefix + "_" + filename))
