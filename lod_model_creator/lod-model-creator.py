# TODO refactoring needed

from typing import Optional
from natsort import natsorted
import os
import shutil
import math
from dataclasses import dataclass

import sys
sys.path.append("..")
from common.Box import Box
from common.ytyp.YtypParser import YtypParser

#TEXTURE_DICTIONARY_PREFIX = "forest"
TEXTURE_DICTIONARY_PREFIX = "vremastered"

# needed to avoid getting wrong pixels from texture (e.g. using 0 would result in getting right-most resp. down-most pixel)
TEXTURE_UV_EPS = 1 / 256

ytypName = TEXTURE_DICTIONARY_PREFIX + "_lod"


@dataclass
class UV:
	u: float
	v: float

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

	def textureOriginSide(self):
		return self.texture_origin if self._textureOriginSide is None else self._textureOriginSide

	def texture(self):
		return "lod_" + self.name.lower()

	def hasDedicatedSideTexture(self):
		return self.uvSideMin is not None and self.uvSideMax is not None

	def getTextureDictionary(self):
		if self.name.startswith("Prop_Bush_"):
			dictMain = "bushes"
		elif self.name.startswith("Prop_Palm_") or self.name.startswith("Prop_Fan_Palm_"):
			dictMain = "palms"
		else:
			dictMain = "trees"
		return TEXTURE_DICTIONARY_PREFIX + "_" + dictMain + "_lod"


trees = {
	# trees
	"Prop_Tree_Birch_01" : Tree("Prop_Tree_Birch_01", 0.546875, 0.6640625, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.7734375, 0.5078125)),
	"Prop_Tree_Birch_02": Tree("Prop_Tree_Birch_02", 0.421875, 0.5703125, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.765625, 0.3671875)),
	"Prop_Tree_Birch_03": Tree("Prop_Tree_Birch_03", 0.546875),
	"Prop_Tree_Birch_03b": Tree("Prop_Tree_Birch_03b", 0.5625),
	"Prop_Tree_Birch_04": Tree("Prop_Tree_Birch_04", 0.5625, 0.421875, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0), UV(0.7734375, 0.453125)),
	"Prop_Tree_Maple_02": Tree("Prop_Tree_Maple_02", 0.4375),
	"Prop_Tree_Maple_03": Tree("Prop_Tree_Maple_03", 0.5),
	"Prop_Tree_Cedar_02": Tree("Prop_Tree_Cedar_02", 0.515625, 0.46315789473, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
	"Prop_Tree_Cedar_03": Tree("Prop_Tree_Cedar_03", 0.5390625, 0.51052631578, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
	"Prop_Tree_Cedar_04": Tree("Prop_Tree_Cedar_04", 0.484375, 0.3947368421, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1), UV(0.484375, 0.87890625)),
	"Prop_Tree_Cedar_S_01": Tree("Prop_Tree_Cedar_S_01", 0.484375, 0.71875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1), UV(0.46875, 0.8203125)),
	"Prop_Tree_Cedar_S_04": Tree("Prop_Tree_Cedar_S_04", 0.5, 0.68269230769, UV(0, 0), UV(1, 0.8125), UV(0, 1), UV(1, 0.8125)),
	"Prop_Tree_Cedar_S_05": Tree("Prop_Tree_Cedar_S_05", 0.46875),
	"Prop_Tree_Cedar_S_06": Tree("Prop_Tree_Cedar_S_06", 0.5),
	"Prop_Tree_Cypress_01": Tree("Prop_Tree_Cypress_01", 0.5),
	"Prop_Tree_Eng_Oak_01": Tree("Prop_Tree_Eng_Oak_01", 0.5, 0.5703125, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.75, 0.53125)),
	"Prop_Tree_Eucalip_01": Tree("Prop_Tree_Eucalip_01", 0.5, 0.359375, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1)),
	"Prop_Tree_Jacada_01": Tree("Prop_Tree_Jacada_01", 0.484375, 0.46875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
	"Prop_Tree_Jacada_02": Tree("Prop_Tree_Jacada_02", 0.515625, 0.546875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
	"Prop_Tree_Oak_01": Tree("Prop_Tree_Oak_01", 0.4765625, 0.4765625, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1), UV(0.46875, 0.75)),
	"Prop_Tree_Olive_01": Tree("Prop_Tree_Olive_01", 0.5, 0.40625, UV(0, 0), UV(1, 0.5), UV(0, 1), UV(1, 0.5)),
	"Prop_Tree_Pine_01": Tree("Prop_Tree_Pine_01", 0.515625, 0.515625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.515625, 0.79296875)),
	"Prop_Tree_Pine_02": Tree("Prop_Tree_Pine_02", 0.546875, 0.6875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.5, 0.80078125)),
	"Prop_Tree_Fallen_Pine_01": Tree("Prop_Tree_Fallen_Pine_01", 0.609375, 1, UV(0, 0), UV(1, 0.625)), #, UV(0, 1), UV(1, 0.625), UV(0.390625, 0.7734375)),
	"Prop_S_Pine_Dead_01": Tree("Prop_S_Pine_Dead_01", 0.40625, 0.4875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.53125, 0.8515625)),
	"Prop_W_R_Cedar_01": Tree("Prop_W_R_Cedar_01", 0.5, 0.8, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
	"Prop_W_R_Cedar_Dead": Tree("Prop_W_R_Cedar_Dead", 0.59375, 0.425, UV(0, 0), UV(1, 0.625) , UV(0, 1), UV(1, 0.625), UV(0.53125, 0.78125)),
	"TEST_Tree_Cedar_Trunk_001": Tree("TEST_Tree_Cedar_Trunk_001", 0.515625, 0.5769231, UV(0, 0), UV(1, 0.8125), UV(0, 1), UV(1, 0.8125)),
	"TEST_Tree_Forest_Trunk_01": Tree("TEST_Tree_Forest_Trunk_01", 0.515625, 0.3894231, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1)),
	"TEST_Tree_Forest_Trunk_04": Tree("TEST_Tree_Forest_Trunk_04", 0.453125, 1, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0), UV(0.78125, 0.5546875), 0.6484375),
	# trees2
	# TODO provide more slod models:
	# prop_rio_del_01
	"Prop_Tree_LFicus_02": Tree("Prop_Tree_LFicus_02", 0.4453125, 0.55, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
	"Prop_Tree_LFicus_03": Tree("Prop_Tree_LFicus_03", 0.46875, 0.359375, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
	"Prop_Tree_LFicus_05": Tree("Prop_Tree_LFicus_05", 0.46875, 0.3125, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
	"Prop_Tree_LFicus_06": Tree("Prop_Tree_LFicus_06", 0.453125, 0.43, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1), None, 0.15),
	# bushes
	"Prop_Bush_Lrg_04b": Tree("Prop_Bush_Lrg_04b", 0.37333333, 0.40625, UV(0, 0), UV(0.5859375, 0.5), UV(0, 0.5), UV(1, 1), None, None, UV(0.5859375, 0), UV(1, 0.5), 0.45283018867),
	"Prop_Bush_Lrg_04c": Tree("Prop_Bush_Lrg_04c", 0.37333333, 0.4375, UV(0, 0), UV(0.5859375, 0.5), UV(0, 0.5), UV(1, 1), None, None, UV(0.5859375, 0), UV(1, 0.5), 0.5660377358),
	"Prop_Bush_Lrg_04d": Tree("Prop_Bush_Lrg_04d", 0.4, 0.421875, UV(0, 0), UV(0.5078125, 0.5), UV(0, 0.5), UV(0.75, 1), None, None, UV(0.5078125, 0), UV(1, 0.484375), 0.47619047619),
	# palms
	"Prop_Palm_Fan_02_b": Tree("Prop_Palm_Fan_02_b", 0.515625, 0.25625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.484375, 0.8125)),
	"Prop_Palm_Fan_03_c": Tree("Prop_Palm_Fan_03_c", 0.5, 0.166666667, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
	"Prop_Palm_Fan_03_d": Tree("Prop_Palm_Fan_03_d", 0.484375, 0.15865384615, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1), UV(0.484375, 0.90625)),
	"Prop_Palm_Fan_04_b": Tree("Prop_Palm_Fan_04_b", 0.484375, 0.321875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1), UV(0.46875, 0.80859375)),
	"Prop_Palm_Huge_01a": Tree("Prop_Palm_Huge_01a", 0.46875, 0.11574074074, UV(0, 0), UV(1, 0.84375), UV(0, 1), UV(1, 0.84375), UV(0.484375, 0.921875)),
	"Prop_Palm_Med_01b": Tree("Prop_Palm_Med_01b", 0.515625, 0.24431818181, UV(0, 0), UV(1, 0.6875), UV(0, 1), UV(1, 0.6875), UV(0.546875, 0.84375)),
	"Prop_Palm_Sm_01e": Tree("Prop_Palm_Sm_01e", 0.515625, None, UV(0, 0), UV(1, 1)),
}
# add other Props that should use the same UV mapping
trees["Prop_Palm_Sm_01d"] = trees["Prop_Palm_Sm_01f"] = \
	trees["Prop_Palm_Sm_01e"]
trees["Prop_Palm_Med_01a"] = trees["Prop_Palm_Med_01c"] = \
	trees["Prop_Palm_Med_01b"]
trees["Prop_Fan_Palm_01a"] = trees["Prop_Palm_Fan_02_a"] = \
	trees["Prop_Palm_Fan_02_b"]
trees["Prop_Palm_Sm_01a"] = trees["Prop_Palm_Fan_04_a"] = trees["Prop_Palm_Fan_04_b"] = \
	trees["Prop_Palm_Fan_04_b"]
trees["Prop_Palm_Fan_03_a"] = trees["Prop_Palm_Fan_03_b"] = trees["Prop_Palm_Fan_03_c_graff"] = trees["Prop_Palm_Fan_04_c"] = \
	trees["Prop_Palm_Fan_03_c"]
trees["Prop_Palm_Med_01d"] = trees["Prop_Palm_Fan_03_d_Graff"] = trees["Prop_Palm_Fan_04_d"] = \
	trees["Prop_Palm_Fan_03_d"]
trees["Prop_Palm_Huge_01b"] = \
	trees["Prop_Palm_Huge_01a"]



template2PlanesLodOdr = open(os.path.join(os.path.dirname(__file__), 'templates','template_2_planes_lod.odr'), 'r').read()
template2PlanesLodMesh = open(os.path.join(os.path.dirname(__file__), 'templates','template_2_planes_lod.mesh'), 'r').read()
template3PlanesLodOdr = open(os.path.join(os.path.dirname(__file__), 'templates','template_3_planes_lod.odr'), 'r').read()
template3PlanesLodMesh = open(os.path.join(os.path.dirname(__file__), 'templates','template_3_planes_lod.mesh'), 'r').read()
templateLodYtypItem = open(os.path.join(os.path.dirname(__file__), 'templates','template_lod_ytyp_item.xml'), 'r').read()


ytyps = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "..", "resources", "ytyp"))


def createTextureUvWithEps(minUv, maxUv):
	if minUv is None or maxUv is None:
		return minUv, maxUv

	minUvEps = UV(minUv.u, minUv.v)
	maxUvEps = UV(maxUv.u, maxUv.v)
	if minUvEps.u < maxUvEps.u:
		minUvEps.u += TEXTURE_UV_EPS
	else:
		maxUvEps.u += TEXTURE_UV_EPS

	if minUvEps.v < maxUvEps.v:
		minUvEps.v += TEXTURE_UV_EPS
	else:
		maxUvEps.v += TEXTURE_UV_EPS

	return minUvEps, maxUvEps


def replacePlaceholders(template: str, propName: str, tree: Tree, bbox: Box, planeIntersection: list[float]):
	bsphere = bbox.getEnclosingSphere()
	bboxSize = bbox.getSizes()

	uvFrontMin, uvFrontMax = createTextureUvWithEps(tree.uvFrontMin, tree.uvFrontMax)
	uvTopMin, uvTopMax = createTextureUvWithEps(tree.uvTopMin, tree.uvTopMax)
	uvSideMin, uvSideMax = createTextureUvWithEps(tree.uvSideMin, tree.uvSideMax)
	if uvSideMin is None:
		uvSideMin = uvFrontMin
	if uvSideMax is None:
		uvSideMax = uvFrontMax

	result = template \
		.replace("${TEXTURE_DICTIONARY}", tree.getTextureDictionary()) \
		.replace("${PROP_NAME.NORMAL}", propName) \
		.replace("${PROP_NAME.LOWERCASE}", propName.lower()) \
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
		.replace("${math.sqrt(BBOX.SIZE.X**2 + BBOX.SIZE.Z**2) / 2}", str(math.sqrt(bboxSize[0]**2 + bboxSize[2]**2) / 2)) \
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
			#planeMaxZ = min(bboxMin[2] + (bboxMax[2] - bboxMin[2]) * 0.9, planeMinZ + (bboxMax[0] - bboxMin[0]) / 2)
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


def createLodModel(prop: str, tree: Tree, bbox: Box):
	global generatedDir, ytypItems

	sizes = bbox.getSizes()

	sizeX = sizes[0]
	sizeY = sizes[1]
	planeIntersection = [bbox.min[0] + sizeX * tree.texture_origin, bbox.min[1] + sizeY * tree.textureOriginSide()]

	#if tree.hasDedicatedSideTexture():
	#	sizeX = bboxMax[0] - bboxMin[0]
	#	centeredX = bboxMin[0] + sizeX * tree.texture_origin
	#	bboxMin[0] -= centeredX
	#	bboxMax[0] -= centeredX
	#else:
	#	 TODO Fehler bei Huge Palm
	#	scaleX = min(bboxMax[0] / (bboxMax[1] - centeredY), bboxMin[0] / (bboxMin[1] + centeredY))
	#	bboxMin[0] = scaleX * (bboxMin[1] - centeredY)
	#	bboxMax[0] = scaleX * (bboxMax[1] - centeredY)

	if tree.uvTopMin is None or tree.uvTopMax is None:
		templateOdr = template2PlanesLodOdr
		templateMesh = template2PlanesLodMesh
	else:
		templateOdr = template3PlanesLodOdr
		templateMesh = template3PlanesLodMesh

	open(os.path.join(generatedDir, "models", prop.lower() + "_lod.odr"), 'w') \
		.write(replacePlaceholders(templateOdr, prop, tree, bbox, planeIntersection))

	open(os.path.join(generatedDir, "meshes", prop.lower() + "_lod.mesh"), 'w') \
		.write(replacePlaceholders(templateMesh, prop, tree, bbox, planeIntersection))

	ytypItems.write(replacePlaceholders(templateLodYtypItem, prop, tree, bbox, planeIntersection))


generatedDir = os.path.join(os.path.dirname(__file__), "generated")
if os.path.exists(generatedDir):
	shutil.rmtree(generatedDir)
os.mkdir(generatedDir)
os.mkdir(os.path.join(generatedDir, "meshes"))
os.mkdir(os.path.join(generatedDir, "models"))
ytypItems = open(os.path.join(generatedDir, "models", ytypName + ".ytyp.xml"), 'w' )

ytypItems.write("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<CMapTypes>
  <extensions/>
  <archetypes>
""")

for name in natsorted(trees):
	tree = trees[name]
	ytypItem = ytyps[name]

	createLodModel(name, tree, ytypItem.boundingBox)

ytypItems.write("""  </archetypes>
  <name>""" + ytypName + """</name>
  <dependencies/>
  <compositeEntityTypes/>
</CMapTypes>""")