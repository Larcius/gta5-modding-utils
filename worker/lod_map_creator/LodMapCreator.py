import math
import random
import shutil
import os
import re
import numpy as np
import transforms3d
from numpy.linalg import norm
from re import Match
from typing import IO, Optional
from natsort import natsorted

from common.BoundingGeometry import BoundingGeometry
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
from worker.lod_map_creator.LodCandidate import LodCandidate
from worker.lod_map_creator.Manifest import Manifest


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
    slodYtypItems: Optional[IO]
    slodCandidates: dict[str, UVMap]
    foundLod: bool
    foundSlod: bool

    lodCandidates: dict[str, LodCandidate]

    MAX_NUM_CHILDREN_IN_DRAWABLE_DICTIONARY = 63

    def prepareLodCandidates(self):
        lodCandidates = {
            # trees
            "prop_tree_birch_01": LodCandidate(0.546875, 0.6484375, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.7734375, 0.5078125)),
            "prop_tree_birch_02": LodCandidate(0.421875, 0.4765625, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.765625, 0.3671875)),
            "prop_tree_birch_03": LodCandidate(0.546875),
            "prop_tree_birch_03b": LodCandidate(0.5625),
            "prop_tree_birch_04": LodCandidate(0.5625, 0.3515625, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0), UV(0.7734375, 0.453125)),
            "prop_tree_maple_02": LodCandidate(0.421875),
            "prop_tree_maple_03": LodCandidate(0.5),
            "prop_tree_cedar_02": LodCandidate(0.5078125, 0.40104166666, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_cedar_03": LodCandidate(0.5234375, 0.46875, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_cedar_04": LodCandidate(0.484375, 0.34375, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1), UV(0.484375, 0.87890625)),
            "prop_tree_cedar_s_01": LodCandidate(0.484375, 0.66875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1), UV(0.46875, 0.8203125)),
            "prop_tree_cedar_s_04": LodCandidate(0.5, 0.67307692307, UV(0, 0), UV(1, 0.8125), UV(0, 1), UV(1, 0.8125)),
            "prop_tree_cedar_s_05": LodCandidate(0.46875),
            "prop_tree_cedar_s_06": LodCandidate(0.5),
            "prop_tree_cypress_01": LodCandidate(0.5, 0.66666666666, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_eng_oak_01": LodCandidate(0.5, 0.375, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.75, 0.53125)),
            "prop_tree_eucalip_01": LodCandidate(0.5, 0.28125, UV(0, 0), UV(0.3125, 1), UV(0.65625, 0), UV(1, 1), None, UV(0.3125, 0), UV(0.65625, 1), 0.5),
            "prop_tree_jacada_01": LodCandidate(0.484375, 0.421875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_jacada_02": LodCandidate(0.515625, 0.34375, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_oak_01": LodCandidate(0.46875, 0.453125, UV(0, 0), UV(1, 0.328125), UV(0, 0.65625), UV(1, 1), None, UV(0, 0.328125), UV(1, 0.65625), 0.5078125),
            "prop_tree_olive_01": LodCandidate(0.5, 0.375, UV(0, 0), UV(1, 0.5), UV(0, 1), UV(1, 0.5)),
            "prop_tree_pine_01": LodCandidate(0.515625, 0.50625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.515625, 0.79296875)),
            "prop_tree_pine_02": LodCandidate(0.546875, 0.63125, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.5, 0.80078125)),
            "prop_tree_fallen_pine_01": LodCandidate(0.609375, None, UV(0, 0), UV(1, 1)),
            "prop_s_pine_dead_01": LodCandidate(0.40625, 0.4875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.53125, 0.8515625)),
            "prop_w_r_cedar_01": LodCandidate(0.515625, 0.67708333333, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_w_r_cedar_dead": LodCandidate(0.59375, 0.425, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.53125, 0.78125)),
            "test_tree_cedar_trunk_001": LodCandidate(0.5234375, 0.54807692307, UV(0, 0), UV(1, 0.8125), UV(0, 1), UV(1, 0.8125)),
            "test_tree_forest_trunk_01": LodCandidate(0.515625, 0.38461538461, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1)),
            "test_tree_forest_trunk_04": LodCandidate(0.453125),
            # trees2
            "prop_tree_lficus_02": LodCandidate(0.4453125, 0.359375, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_03": LodCandidate(0.46875, 0.21875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_05": LodCandidate(0.46875, 0.203125, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_06": LodCandidate(0.453125, 0.25, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_mquite_01": LodCandidate(0.46875),
            "prop_rio_del_01": LodCandidate(0.53125),
            "prop_rus_olive": LodCandidate(0.484375, 0.546875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_rus_olive_wint": LodCandidate(0.556),
            # bushes
            "prop_bush_lrg_04b": LodCandidate(0.375, 0.46875, UV(0, 0), UV(0.59375, 0.5), UV(0, 0.5), UV(1, 1), None, UV(0.59375, 0), UV(1, 0.5), 0.4423076923),
            "prop_bush_lrg_04c": LodCandidate(0.38157894736, 0.453125, UV(0, 0), UV(0.59375, 0.5), UV(0, 0.5), UV(1, 1), None, UV(0.59375, 0), UV(1, 0.5), 0.4423076923),
            "prop_bush_lrg_04d": LodCandidate(0.38970588235, 0.484375, UV(0, 0), UV(0.53125, 0.5), UV(0, 0.5), UV(0.75, 1), None, UV(0.53125, 0), UV(1, 0.5), 0.5),
            # palms
            "prop_palm_fan_02_b": LodCandidate(0.515625, 0.13125, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625), UV(0.484375, 0.8125)),
            "prop_palm_fan_03_c": LodCandidate(0.5, 0.08854166666, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_palm_fan_03_d": LodCandidate(0.484375, 0.08173076923, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1), UV(0.484375, 0.90625)),
            "prop_palm_fan_04_b": LodCandidate(0.484375, 0.20625, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1), UV(0.46875, 0.80859375)),
            "prop_palm_fan_04_c": LodCandidate(0.5, 0.140625, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_palm_fan_04_d": LodCandidate(0.421875, 0.12019230769, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1)),
            "prop_palm_huge_01a": LodCandidate(0.484375, 0.05092592592, UV(0, 0), UV(1, 0.84375), UV(0, 1), UV(1, 0.84375), UV(0.484375, 0.921875), UV(0, 0), UV(1, 70/512), None, (432-70)/432),
            "prop_palm_huge_01b": LodCandidate(0.4765625, 0.04166666666, UV(0, 0), UV(1, 0.84375), UV(0, 1), UV(1, 0.84375), UV(0.484375, 0.921875), UV(0, 0), UV(1, 67/512), None, (432-67)/432),
            "prop_palm_med_01b": LodCandidate(0.515625, 0.17613636363, UV(0, 0), UV(1, 0.6875), UV(0, 1), UV(1, 0.6875), UV(0.546875, 0.84375)),
            "prop_palm_med_01c": LodCandidate(0.515625, 0.16666666666, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_palm_med_01d": LodCandidate(0.5, 0.11057692307, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1)),
            "prop_palm_sm_01a": LodCandidate(0.46875, None, UV(0, 0), UV(1, 1)),
            "prop_palm_sm_01d": LodCandidate(0.515625, None, UV(0, 0), UV(1, 1)),
            "prop_palm_sm_01e": LodCandidate(0.515625, None, UV(0, 0), UV(1, 1)),
            "prop_palm_sm_01f": LodCandidate(0.515625, None, UV(0, 0), UV(1, 1)),
            "prop_veg_crop_tr_01": LodCandidate(0.484375, None, UV(0, 0), UV(1, 1)),
            "prop_veg_crop_tr_02": LodCandidate(0.5, None, UV(0, 0), UV(1, 1))
        }
        # for each lodCandidate set its diffuseSampler as lod_<archetype name>
        for name in lodCandidates:
            lodCandidates[name].setDiffuseSampler(name)

        # add other Props that should use the same UV mapping
        lodCandidates["prop_palm_med_01a"] = lodCandidates["prop_palm_med_01b"]
        lodCandidates["prop_fan_palm_01a"] = lodCandidates["prop_palm_fan_02_a"] = lodCandidates["prop_palm_fan_02_b"]
        lodCandidates["prop_palm_fan_04_a"] = lodCandidates["prop_palm_fan_04_b"]
        lodCandidates["prop_palm_fan_03_a"] = lodCandidates["prop_palm_fan_03_b"] = lodCandidates["prop_palm_fan_03_c_graff"] = \
            lodCandidates["prop_palm_fan_03_c"]
        lodCandidates["prop_palm_fan_03_d_graff"] = lodCandidates["prop_palm_fan_03_d"]

        self.lodCandidates = lodCandidates

    def prepareSlodCandidates(self):
        # ensure that lodCandidates are prepared
        if self.lodCandidates is None:
            self.prepareLodCandidates()

        slodCandidates = {
            # trees2
            "prop_tree_lficus_02": UVMap("trees2", UV(0 / 2, 0 / 4), UV(1 / 2, 1 / 4), UV(0 / 2, 0 / 4), UV(1 / 2, 1 / 4)),
            "prop_tree_lficus_03": UVMap("trees2", UV(1 / 2, 0 / 4), UV(2 / 2, 1 / 4), UV(1 / 2, 0 / 4), UV(2 / 2, 1 / 4)),
            "prop_tree_lficus_05": UVMap("trees2", UV(0 / 2, 1 / 4), UV(1 / 2, 2 / 4), UV(0 / 2, 1 / 4), UV(1 / 2, 2 / 4)),
            "prop_tree_lficus_06": UVMap("trees2", UV(1 / 2, 1 / 4), UV(2 / 2, 2 / 4), UV(1 / 2, 1 / 4), UV(2 / 2, 2 / 4)),
            "prop_tree_mquite_01": UVMap("trees2", UV(0 / 2, 2 / 4), UV(1 / 2, 3 / 4)),
            "prop_rio_del_01": UVMap("trees2", UV(1 / 2, 2 / 4), UV(2 / 2, 3 / 4)),
            "prop_rus_olive": UVMap("trees2", UV(0 / 2, 3 / 4), UV(1 / 2, 4 / 4), UV(0 / 2, 2 / 4), UV(1 / 2, 3 / 4)),
            "prop_rus_olive_wint": UVMap("trees2", UV(1 / 2, 3 / 4), UV(2 / 2, 4 / 4)),
            # trees
            "prop_s_pine_dead_01": UVMap("trees", UV(10 / 16, 6 / 16), UV(12 / 16, 11 / 16), UV(0 / 16, 12 / 16), UV(3 / 16, 14 / 16)),
            "prop_tree_birch_01": UVMap("trees", UV(8 / 16, 6 / 16), UV(10 / 16, 10 / 16), UV(0 / 16, 3 / 16), UV(3 / 16, 6 / 16)),
            "prop_tree_birch_02": UVMap("trees", UV(6 / 16, 3 / 16), UV(9 / 16, 6 / 16), UV(3 / 16, 3 / 16), UV(6 / 16, 6 / 16)),
            "prop_tree_birch_03": UVMap("trees", UV(4 / 16, 3 / 16), UV(6 / 16, 6 / 16)),
            "prop_tree_birch_03b": UVMap("trees", UV(2 / 16, 3 / 16), UV(4 / 16, 5 / 16)),
            "prop_tree_birch_04": UVMap("trees", UV(9 / 16, 3 / 16), UV(12 / 16, 6 / 16), UV(6 / 16, 3 / 16), UV(9 / 16, 6 / 16)),
            "prop_tree_cedar_02": UVMap("trees", UV(4 / 16, 11 / 16), UV(6 / 16, 16 / 16), UV(6 / 16, 6 / 16), UV(9 / 16, 9 / 16)),
            "prop_tree_cedar_03": UVMap("trees", UV(6 / 16, 10 / 16), UV(8 / 16, 16 / 16), UV(9 / 16, 6 / 16), UV(12 / 16, 9 / 16)),
            "prop_tree_cedar_04": UVMap("trees", UV(8 / 16, 10 / 16), UV(10 / 16, 16 / 16), UV(12 / 16, 6 / 16), UV(15 / 16, 9 / 16)),
            "prop_tree_cedar_s_01": UVMap("trees", UV(6 / 16, 6 / 16), UV(8 / 16, 10 / 16)),
            "prop_tree_cedar_s_04": UVMap("trees", UV(3 / 16, 8 / 16), UV(4 / 16, 12 / 16), UV(3 / 16, 6 / 16), UV(3 / 16, 6 / 16)),
            "prop_tree_cedar_s_05": UVMap("trees", UV(3 / 16, 8 / 16), UV(4 / 16, 12 / 16)),
            "prop_tree_cedar_s_06": UVMap("trees", UV(6 / 16, 0 / 16), UV(7 / 16, 3 / 16)),
            "prop_tree_cypress_01": UVMap("trees", UV(4 / 16, 6 / 16), UV(6 / 16, 11 / 16)),
            "prop_tree_eng_oak_01": UVMap("trees", UV(10 / 16, 0 / 16), UV(13 / 16, 3 / 16), UV(9 / 16, 0 / 16), UV(12 / 16, 3 / 16)),
            "prop_tree_eucalip_01": UVMap("trees", UV(0 / 16, 8 / 16), UV(3 / 16, 12 / 16), UV(9 / 16, 3 / 16), UV(12 / 16, 6 / 16)),
            "prop_tree_fallen_pine_01": UVMap("trees", UV(12 / 16, 3 / 16), UV(14 / 16, 8 / 16)),
            "prop_tree_jacada_01": UVMap("trees", UV(0 / 16, 0 / 16), UV(3 / 16, 3 / 16), UV(0 / 16, 0 / 16), UV(3 / 16, 3 / 16)),
            "prop_tree_jacada_02": UVMap("trees", UV(3 / 16, 0 / 16), UV(6 / 16, 3 / 16), UV(3 / 16, 0 / 16), UV(6 / 16, 3 / 16)),
            "prop_tree_maple_02": UVMap("trees", UV(0 / 16, 3 / 16), UV(2 / 16, 5 / 16)),
            "prop_tree_maple_03": UVMap("trees", UV(0 / 16, 5 / 16), UV(2 / 16, 8 / 16)),
            "prop_tree_oak_01": UVMap("trees", UV(13 / 16, 0 / 16), UV(16 / 16, 3 / 16), UV(12 / 16, 0 / 16), UV(15 / 16, 3 / 16)),
            "prop_tree_olive_01": UVMap("trees", UV(7 / 16, 0 / 16), UV(10 / 16, 3 / 16), UV(6 / 16, 0 / 16), UV(9 / 16, 3 / 16)),
            "prop_tree_pine_01": UVMap("trees", UV(0 / 16, 12 / 16), UV(2 / 16, 16 / 16), UV(0 / 16, 9 / 16), UV(3 / 16, 12 / 16)),
            "prop_tree_pine_02": UVMap("trees", UV(2 / 16, 12 / 16), UV(4 / 16, 16 / 16), UV(3 / 16, 9 / 16), UV(6 / 16, 12 / 16)),
            "prop_w_r_cedar_01": UVMap("trees", UV(10 / 16, 11 / 16), UV(12 / 16, 16 / 16), UV(6 / 16, 9 / 16), UV(9 / 16, 12 / 16)),
            "prop_w_r_cedar_dead": UVMap("trees", UV(14 / 16, 3 / 16), UV(16 / 16, 8 / 16), UV(3 / 16, 12 / 16), UV(6 / 16, 15 / 16)),
            "test_tree_cedar_trunk_001": UVMap("trees", UV(12 / 16, 8 / 16), UV(14 / 16, 16 / 16), UV(9 / 16, 9 / 16), UV(12 / 16, 12 / 16)),
            "test_tree_forest_trunk_01": UVMap("trees", UV(14 / 16, 8 / 16), UV(16 / 16, 16 / 16), UV(12 / 16, 9 / 16), UV(15 / 16, 12 / 16)),
            "test_tree_forest_trunk_04": UVMap("trees", UV(2 / 16, 5 / 16), UV(4 / 16, 8 / 16)),
            # bushes
            "prop_bush_lrg_04b": UVMap("bushes", UV(0.5, 0), UV(1, 0.5), UV(0.5, 0), UV(1, 0.5)),
            "prop_bush_lrg_04c": UVMap("bushes", UV(0, 0.5), UV(0.5, 1), UV(0, 0.5), UV(0.5, 1)),
            "prop_bush_lrg_04d": UVMap("bushes", UV(0.5, 0.5), UV(1, 1), UV(0.5, 0.5), UV(1, 1)),
            # palms
            "prop_palm_sm_01e": UVMap("palms", UV(0 / 4, 0 / 4), UV(1 / 4, 2 / 4)),
            "prop_palm_fan_02_b": UVMap("palms", UV(0 / 4, 2 / 4), UV(1 / 4, 4 / 4), UV(0, 0), UV(0.5, 0.5), 0.23692810457),
            "prop_palm_fan_03_c": UVMap("palms", UV(1 / 4, 0 / 4), UV(2 / 4, 4 / 4), UV(0.5, 0), UV(1, 0.5), 0.14356435643),
            "prop_palm_fan_03_d": UVMap("palms", UV(2 / 4, 0 / 4), UV(3 / 4, 4 / 4), UV(0, 0.5), UV(0.5, 1), 0.13046937151),
            "prop_palm_huge_01a": UVMap("palms", UV(3 / 4, 0 / 4), UV(4 / 4, 4 / 4), UV(0.5, 0.5), UV(1, 1), 0.09644268774)
        }
        # for each missing topZ in slodCandidates get planeZ from lodCandidates and set that value
        for name in slodCandidates:
            if slodCandidates[name].topZ is None:
                slodCandidates[name].topZ = self.lodCandidates[name].planeZ

        # add other Props that should use the same UV mapping
        slodCandidates["prop_palm_sm_01d"] = slodCandidates["prop_palm_sm_01f"] = slodCandidates["prop_palm_med_01a"] = slodCandidates["prop_palm_med_01b"] = \
            slodCandidates["prop_palm_med_01c"] = slodCandidates["prop_palm_sm_01e"]
        slodCandidates["prop_fan_palm_01a"] = slodCandidates["prop_palm_fan_02_a"] = slodCandidates["prop_palm_sm_01a"] = \
            slodCandidates["prop_palm_fan_04_a"] = slodCandidates["prop_palm_fan_04_b"] = slodCandidates["prop_palm_fan_02_b"]
        slodCandidates["prop_palm_fan_03_a"] = slodCandidates["prop_palm_fan_03_b"] = slodCandidates["prop_palm_fan_03_c_graff"] = \
            slodCandidates["prop_palm_fan_04_c"] = slodCandidates["prop_palm_fan_03_c"]
        slodCandidates["prop_palm_med_01d"] = slodCandidates["prop_palm_fan_03_d_graff"] = slodCandidates["prop_palm_fan_04_d"] = \
            slodCandidates["prop_palm_fan_03_d"]
        slodCandidates["prop_palm_huge_01b"] = \
            slodCandidates["prop_palm_huge_01a"]

        self.slodCandidates = slodCandidates

    LOD_DISTANCE = 750
    SLOD_DISTANCE = 1100  # somehow arbitrary and not that important because SLOD1 and SLOD2 are almost the same models.
    # however, reducing this results in smaller streaming extents of LOD/SLOD1 maps
    SLOD2_DISTANCE = 2800  # using 2800 because max height in game is 2600 and therefore until that height (plus a bit to allow slight xy offset)
    # a model with xy plane is needed so that objects don't vanish when above (SLOD2 models do contain such a xy plane but not SLOD3)
    SLOD3_DISTANCE = 15000  # that seems to be the default value from Rockstars (in fact the whole map is not that large anyway)

    NUM_CHILDREN_MAX_VALUE = 255  # TODO confirm following claim: must be <= 255 since numChildren is of size 1 byte
    LOD_DISTANCES_MAX_DIFFERENCE_LOD = 60
    ENTITIES_EXTENTS_MAX_DIAGONAL_LOD = 100
    ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD1 = 400
    ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD2 = 800
    ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD3 = 1600

    USE_SLOD_AS_LOD_MODEL = False

    unitBox = Box.createUnitBox()
    unitSphere = Sphere.createUnitSphere()

    # only entities with a lodDistance (according to hd entity) greater or equal this value are considered for SLOD1 to 3 model
    MIN_HD_LOD_DISTANCE_FOR_SLOD1 = Util.calculateLodDistance(unitBox, unitSphere, [4] * 3, True)  # 180
    MIN_HD_LOD_DISTANCE_FOR_SLOD2 = Util.calculateLodDistance(unitBox, unitSphere, [8] * 3, True)  # 240
    MIN_HD_LOD_DISTANCE_FOR_SLOD3 = Util.calculateLodDistance(unitBox, unitSphere, [13] * 3, True)  # 290

    def __init__(self, inputDir: str, outputDir: str, prefix: str):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix
        self.slodYtypItems = None
        self.foundLod = False
        self.foundSlod = False

    def run(self):
        print("running lod map creator...")
        self.determinePrefixBundles()
        self.readTemplates()
        self.prepareLodCandidates()
        self.prepareSlodCandidates()
        self.createOutputDir()
        self.readYtypItems()
        self.processFiles()
        self.addLodAndSlodModelsToYtypDict()
        self.fixMapExtents()
        self.createManifest()
        self.copyOthers()
        self.copyTextureDictionaries()
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
        return os.path.join(self.outputDir, "_slod_meshes")

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

        f = open(os.path.join(templatesDir, "slod", "template_shader_tree_lod2.odr.part"), 'r')
        self.contentTemplateOdrShaderTreeLod2 = f.read()
        f.close()

    def readYtypItems(self):
        self.ytypItems = YtypParser.readYtypDirectory(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "ytyp"))

    def replaceFlagsAndContentFlags(self, content: str, flags: int, contentFlags: int) -> str:
        # TODO deal with existing flags, e.g. "Scripted (1)"
        return re.sub(
            '(<flags\\s+value=")[^"]*("\\s*/>\\s*<contentFlags\\s+value=")[^"]*("\\s*/>)',
            "\\g<1>" + str(flags) + "\\g<2>" + str(contentFlags) + "\\g<3>", content
        )

    def replParentIndex(self, matchobj: Match, mutableIndex: list[int], hdToLod: dict[int, int], offsetParentIndex: int) -> str:
        archetypeName = matchobj.group(2).lower()
        if (not self.USE_SLOD_AS_LOD_MODEL and archetypeName in self.lodCandidates) or (self.USE_SLOD_AS_LOD_MODEL and archetypeName in self.slodCandidates):
            index = mutableIndex[0]
            parentIndex = hdToLod[index] + offsetParentIndex
            mutableIndex[0] += 1
        else:
            parentIndex = -1

        return matchobj.group(1) + str(parentIndex) + matchobj.group(3)

    def replacePlaceholders(self, template: str, name: str, textureDictionary: str, drawableDictionary: str, bbox: Box, bsphere: Sphere, hdDistance: float, lodDistance: float) -> str:
        return template \
            .replace("${NAME}", name) \
            .replace("${TEXTURE_DICTIONARY}", textureDictionary) \
            .replace("${DRAWABLE_DICTIONARY}", drawableDictionary) \
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

    def createIndicesForRectangles(self, numRectangles: int) -> list[int]:
        indices = []
        for i in range(numRectangles):
            offset = 4 * i
            indices.append(offset)
            indices.append(offset + 1)
            indices.append(offset + 2)
            indices.append(offset + 2)
            indices.append(offset + 3)
            indices.append(offset)
        return indices

    def createAabb(self, bbox: Box) -> str:
        return self.contentTemplateMeshAabb \
            .replace("${BBOX.MIN.X}", Util.floatToStr(bbox.min[0])) \
            .replace("${BBOX.MIN.Y}", Util.floatToStr(bbox.min[1])) \
            .replace("${BBOX.MIN.Z}", Util.floatToStr(bbox.min[2])) \
            .replace("${BBOX.MAX.X}", Util.floatToStr(bbox.max[0])) \
            .replace("${BBOX.MAX.Y}", Util.floatToStr(bbox.max[1])) \
            .replace("${BBOX.MAX.Z}", Util.floatToStr(bbox.max[2]))

    @staticmethod
    def findClosestRatio(ratioInput: float, ratioCandidate1: float, ratioCandidate2: float) -> (float, int):
        options = [ratioCandidate1, ratioCandidate2]
        argmin = np.abs(np.asarray(options) - ratioInput).argmin()
        return options[argmin], argmin

    @staticmethod
    def createLodModelVertexNormalTextureUVStr(vertex: list[float], normal: list[float], uv: list[float]):
        return "				" + Util.vectorToStr(vertex) + " / " + Util.vectorToStr(Util.normalize(normal)) + " / 255 29 0 255 / " + Util.vectorToStr(uv) + "\n"

    def createIndicesStr(self, indices: list[int]) -> str:
        indicesStr = ""
        i = 0
        for index in indices:
            if i % 15 == 0:
                if i > 0:
                    indicesStr += "\n"
                indicesStr += "				"
            else:
                indicesStr += " "
            indicesStr += str(index)
            i += 1

        return indicesStr

    @staticmethod
    def appendFrontPlaneIndicesForLod(indices: list[int], offset: int):
        indicesTemplate = [0, 1, 2, 0, 1, 2, 2, 3, 0, 2, 3, 0, 4, 5, 6, 4, 5, 6, 6, 7, 4, 6, 7, 4, 8, 11, 10, 8, 11, 10, 10, 9, 8, 10, 9, 8, 12, 15, 14, 12, 15, 14, 14, 13, 12, 14, 13, 12]
        indices += [x + offset for x in indicesTemplate]

    @staticmethod
    def appendTopPlaneIndicesForLod(indices: list[int], offset: int):
        indicesTemplateTop = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1]
        indices += [x + offset for x in indicesTemplateTop]

    def appendFrontPlaneVerticesForLod(self, vertices: list[list[float]], normals: list[list[float]], textureUVs: list[list[float]], entity: EntityItem, planeIntersection: list[float]):
        bbox = self.ytypItems[entity.archetypeName].boundingBox
        height = bbox.getSizes()[2]

        lodCandidate = self.lodCandidates[entity.archetypeName]
        uvFrontMin = lodCandidate.uvFrontMin
        uvFrontMax = lodCandidate.uvFrontMax
        uvSideMin = lodCandidate.getUvSideMin()
        uvSideMax = lodCandidate.getUvSideMax()
        sideOffsetZ = lodCandidate.sideOffsetZ
        minZ = bbox.min[2] + height * max(0.0, sideOffsetZ)
        maxZ = bbox.max[2] - height * min(0.0, sideOffsetZ)

        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.min[0], planeIntersection[1], bbox.min[2]], [-1, -1, 0], [uvFrontMin.u, uvFrontMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.max[0], planeIntersection[1], bbox.min[2]], [1, -1, 0], [uvFrontMax.u, uvFrontMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.max[0], planeIntersection[1], bbox.max[2]], [1, -1, 1], [uvFrontMax.u, uvFrontMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.min[0], planeIntersection[1], bbox.max[2]], [-1, -1, 1], [uvFrontMin.u, uvFrontMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.min[1], minZ], [1, -1, 0], [uvSideMin.u, uvSideMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.max[1], minZ], [1, 1, 0], [uvSideMax.u, uvSideMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.max[1], maxZ], [1, 1, 1], [uvSideMax.u, uvSideMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.min[1], maxZ], [1, -1, 1], [uvSideMin.u, uvSideMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.min[0], planeIntersection[1], bbox.min[2]], [-1, 1, 0], [uvFrontMin.u, uvFrontMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.max[0], planeIntersection[1], bbox.min[2]], [1, 1, 0], [uvFrontMax.u, uvFrontMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.max[0], planeIntersection[1], bbox.max[2]], [1, 1, 1], [uvFrontMax.u, uvFrontMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.min[0], planeIntersection[1], bbox.max[2]], [-1, 1, 1], [uvFrontMin.u, uvFrontMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.min[1], minZ], [-1, -1, 0], [uvSideMin.u, uvSideMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.max[1], minZ], [-1, 1, 0], [uvSideMax.u, uvSideMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.max[1], maxZ], [-1, 1, 1], [uvSideMax.u, uvSideMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], bbox.min[1], maxZ], [-1, -1, 1], [uvSideMin.u, uvSideMin.v])

    def appendDiagonalPlaneVerticesForLod(self, vertices: list[list[float]], normals: list[list[float]], textureUVs: list[list[float]], entity: EntityItem, planeIntersection: list[float]):
        bbox = self.ytypItems[entity.archetypeName].boundingBox

        lodCandidate = self.lodCandidates[entity.archetypeName]

        uvFrontMin = lodCandidate.uvFrontMin
        uvFrontMax = lodCandidate.uvFrontMax
        uvSideMin = lodCandidate.getUvSideMin()
        uvSideMax = lodCandidate.getUvSideMax()

        distanceLeftToIntersection = planeIntersection[0] - bbox.min[0]
        distanceRightToIntersection = bbox.max[0] - planeIntersection[0]
        distanceBottomToIntersection = planeIntersection[1] - bbox.min[1]
        distanceTopToIntersection = bbox.max[1] - planeIntersection[1]

        vectorRightTop = self.calculateVectorOnEllipseAtDiagonal(distanceRightToIntersection, distanceTopToIntersection, 0)
        vectorLeftBottom = self.calculateVectorOnEllipseAtDiagonal(distanceLeftToIntersection, distanceBottomToIntersection, 2)
        lengthVectorRightTop = norm(vectorRightTop)
        lengthVectorLeftBottom = norm(vectorLeftBottom)
        ratio = lengthVectorRightTop / (lengthVectorLeftBottom + lengthVectorRightTop)
        desiredRatio, option = LodMapCreator.findClosestRatio(ratio, lodCandidate.texture_origin, lodCandidate.textureOriginSide())
        uvDiagonal1Min = uvFrontMin if option == 0 else uvSideMin
        uvDiagonal1Max = uvFrontMax if option == 0 else uvSideMax
        if ratio > desiredRatio:
            adapt = desiredRatio / (1 - desiredRatio) * lengthVectorLeftBottom / lengthVectorRightTop
            vectorRightTop = [vectorRightTop[0] * adapt, vectorRightTop[1] * adapt]
        else:
            adapt = (1 - desiredRatio) / desiredRatio * lengthVectorRightTop / lengthVectorLeftBottom
            vectorLeftBottom = [vectorLeftBottom[0] * adapt, vectorLeftBottom[1] * adapt]

        vectorLeftTop = self.calculateVectorOnEllipseAtDiagonal(distanceLeftToIntersection, distanceTopToIntersection, 1)
        vectorRightBottom = self.calculateVectorOnEllipseAtDiagonal(distanceRightToIntersection, distanceBottomToIntersection, 3)
        lengthVectorLeftTop = norm(vectorLeftTop)
        lengthVectorRightBottom = norm(vectorRightBottom)
        ratio = lengthVectorRightBottom / (lengthVectorLeftTop + lengthVectorRightBottom)
        desiredRatio, option = LodMapCreator.findClosestRatio(ratio, lodCandidate.texture_origin, lodCandidate.textureOriginSide())
        uvDiagonal2Min = uvFrontMin if option == 0 else uvSideMin
        uvDiagonal2Max = uvFrontMax if option == 0 else uvSideMax
        if ratio > desiredRatio:
            adapt = desiredRatio / (1 - desiredRatio) * lengthVectorLeftTop / lengthVectorRightBottom
            vectorRightBottom = [vectorRightBottom[0] * adapt, vectorRightBottom[1] * adapt]
        else:
            adapt = (1 - desiredRatio) / desiredRatio * lengthVectorRightBottom / lengthVectorLeftTop
            vectorLeftTop = [vectorLeftTop[0] * adapt, vectorLeftTop[1] * adapt]

        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightTop[0], planeIntersection[1] + vectorRightTop[1], bbox.min[2]], [0, 1, 0], [uvDiagonal1Min.u, uvDiagonal1Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftBottom[0], planeIntersection[1] + vectorLeftBottom[1], bbox.min[2]], [-1, 0, 0], [uvDiagonal1Max.u, uvDiagonal1Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftBottom[0], planeIntersection[1] + vectorLeftBottom[1], bbox.max[2]], [-1, 0, 1], [uvDiagonal1Max.u, uvDiagonal1Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightTop[0], planeIntersection[1] + vectorRightTop[1], bbox.max[2]], [0, 1, 1], [uvDiagonal1Min.u, uvDiagonal1Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightBottom[0], planeIntersection[1] + vectorRightBottom[1], bbox.min[2]], [1, 0, 0], [uvDiagonal2Min.u, uvDiagonal2Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftTop[0], planeIntersection[1] + vectorLeftTop[1], bbox.min[2]], [0, 1, 0], [uvDiagonal2Max.u, uvDiagonal2Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftTop[0], planeIntersection[1] + vectorLeftTop[1], bbox.max[2]], [0, 1, 1], [uvDiagonal2Max.u, uvDiagonal2Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightBottom[0], planeIntersection[1] + vectorRightBottom[1], bbox.max[2]], [1, 0, 1], [uvDiagonal2Min.u, uvDiagonal2Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightTop[0], planeIntersection[1] + vectorRightTop[1], bbox.min[2]], [1, 0, 0], [uvDiagonal1Min.u, uvDiagonal1Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftBottom[0], planeIntersection[1] + vectorLeftBottom[1], bbox.min[2]], [0, -1, 0], [uvDiagonal1Max.u, uvDiagonal1Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftBottom[0], planeIntersection[1] + vectorLeftBottom[1], bbox.max[2]], [0, -1, 1], [uvDiagonal1Max.u, uvDiagonal1Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightTop[0], planeIntersection[1] + vectorRightTop[1], bbox.max[2]], [1, 0, 1], [uvDiagonal1Min.u, uvDiagonal1Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightBottom[0], planeIntersection[1] + vectorRightBottom[1], bbox.min[2]], [0, -1, 0], [uvDiagonal2Min.u, uvDiagonal2Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftTop[0], planeIntersection[1] + vectorLeftTop[1], bbox.min[2]], [-1, 0, 0], [uvDiagonal2Max.u, uvDiagonal2Max.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorLeftTop[0], planeIntersection[1] + vectorLeftTop[1], bbox.max[2]], [-1, 0, 1], [uvDiagonal2Max.u, uvDiagonal2Min.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0] + vectorRightBottom[0], planeIntersection[1] + vectorRightBottom[1], bbox.max[2]], [0, -1, 1], [uvDiagonal2Min.u, uvDiagonal2Min.v])

    def appendTopPlaneVerticesForLod(self, vertices: list[list[float]], normals: list[list[float]], textureUVs: list[list[float]], entity: EntityItem, planeIntersection: list[float]):
        bbox = self.ytypItems[entity.archetypeName].boundingBox
        sizes = bbox.getSizes()

        lodCandidate = self.lodCandidates[entity.archetypeName]

        uvTopMin = lodCandidate.uvTopMin
        uvTopMax = lodCandidate.uvTopMax
        uvTopCenter = lodCandidate.getUvTopCenter()

        planeTopMinZ = bbox.min[2] + sizes[2] * (1 - lodCandidate.planeZ)
        # planeTopMaxZ = min(bbox.max[2] - min(sizes) * 0.1, planeTopMinZ + min(sizes[0], sizes[1]) / 4)
        planeTopMaxZ = max(bbox.min[2] + min(sizes) * 0.2, planeTopMinZ - 0.15 * min(sizes[0], sizes[1]))

        self.appendVertexForLod(vertices, normals, textureUVs, entity, [planeIntersection[0], planeIntersection[1], planeTopMaxZ], [0, 0, 1], [uvTopCenter.u, uvTopCenter.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.min[0], bbox.min[1], planeTopMinZ], [-1, -1, 0.1], [uvTopMin.u, uvTopMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.max[0], bbox.min[1], planeTopMinZ], [1, -1, 0.1], [uvTopMax.u, uvTopMax.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.max[0], bbox.max[1], planeTopMinZ], [1, 1, 0.1], [uvTopMax.u, uvTopMin.v])
        self.appendVertexForLod(vertices, normals, textureUVs, entity, [bbox.min[0], bbox.max[1], planeTopMinZ], [-1, 1, 0.1], [uvTopMin.u, uvTopMin.v])

    def appendVertexForLod(self, vertices: list[list[float]], normals: list[list[float]], textureUVs: list[list[float]], entity: EntityItem, vertex: list[float], normal: list[float], uv: list[float]):
        vertices.append(entity.applyTransformationTo(vertex))
        normals.append(Util.applyRotation(normal, entity.rotation))
        textureUVs.append(uv)

    @staticmethod
    def convertVerticesNormalsTextureUVsAsStr(vertices: list[list[float]], normals: list[list[float]], textureUVs: list[list[float]], translation: list[float]) -> str:
        result = ""
        for i in range(len(vertices)):
            translatedVertex = np.add(vertices[i], translation).tolist()
            result += LodMapCreator.createLodModelVertexNormalTextureUVStr(translatedVertex, normals[i], textureUVs[i])
        return result

    @staticmethod
    def convertVerticesTextureUVsAsStrForSlod(vertices: list[list[float]], sizes: list[list[float]], textureUVs: list[list[UV]], translation: list[float]) -> str:
        result = ""
        for i in range(len(vertices)):
            translatedVertex = np.add(vertices[i], translation).tolist()

            uvMin = textureUVs[i][0]
            uvMax = textureUVs[i][1]

            # for a bit more variety randomly flip/mirror the texture
            # set a seed here to get the same result for multiple runs and different SLOD levels (therefore do not use translatedVertex)
            seed = Util.hashFloat(vertices[i][0]) ^ Util.hashFloat(vertices[i][1])
            random.seed(a=seed)
            if bool(random.getrandbits(1)):
                temp = uvMin.u
                uvMin.u = uvMax.u
                uvMax.u = temp

            result += LodMapCreator.createSlodModelVertexNormalTextureUVStr(translatedVertex, [-1, -1, 0], sizes[i], [0, 1], [uvMin.u, uvMax.v])
            result += LodMapCreator.createSlodModelVertexNormalTextureUVStr(translatedVertex, [1, -1, 0], sizes[i], [1, 1], [uvMax.u, uvMax.v])
            result += LodMapCreator.createSlodModelVertexNormalTextureUVStr(translatedVertex, [1, -1, 1], sizes[i], [1, 0], [uvMax.u, uvMin.v])
            result += LodMapCreator.createSlodModelVertexNormalTextureUVStr(translatedVertex, [-1, -1, 1], sizes[i], [0, 0], [uvMin.u, uvMin.v])

        return result

    @staticmethod
    def createSlodModelVertexNormalTextureUVStr(center: list[float], normal: list[float], size: list[float], uvPosition: list[float], uv: list[float]):
        colorsFront = "255 29 0 255 / 0 255 0 0"
        return "				" + Util.vectorToStr(center) + " / " + Util.vectorToStr(Util.normalize(normal)) + " / " + colorsFront + " / " + Util.vectorToStr(uvPosition) + " / " + Util.vectorToStr(uv) + " / " + Util.vectorToStr(size) + " / " + Util.vectorToStr([1, 1]) + "\n"

    def createLodModel(self, lodName: str, drawableDictionary: str, entities: list[EntityItem], parentIndex: int, numChildren: int) -> EntityItem:
        diffuseSamplerToVertices = {}
        diffuseSamplerToNormals = {}
        diffuseSamplerToTextureUVs = {}
        diffuseSamplerToIndices = {}

        for entity in entities:
            lodCandidate = self.lodCandidates[entity.archetypeName]
            diffuseSampler = lodCandidate.diffuseSampler
            if diffuseSampler not in diffuseSamplerToVertices:
                diffuseSamplerToVertices[diffuseSampler] = []
                diffuseSamplerToNormals[diffuseSampler] = []
                diffuseSamplerToTextureUVs[diffuseSampler] = []
                diffuseSamplerToIndices[diffuseSampler] = []

            bbox = self.ytypItems[entity.archetypeName].boundingBox
            sizes = bbox.getSizes()
            distanceLeftToIntersection = sizes[0] * lodCandidate.texture_origin
            distanceBottomToIntersection = sizes[1] * lodCandidate.textureOriginSide()
            planeIntersection = [bbox.min[0] + distanceLeftToIntersection, bbox.min[1] + distanceBottomToIntersection]

            LodMapCreator.appendFrontPlaneIndicesForLod(diffuseSamplerToIndices[diffuseSampler], len(diffuseSamplerToVertices[diffuseSampler]))
            self.appendFrontPlaneVerticesForLod(diffuseSamplerToVertices[diffuseSampler], diffuseSamplerToNormals[diffuseSampler], diffuseSamplerToTextureUVs[diffuseSampler], entity, planeIntersection)

            if lodCandidate.hasDiagonal(bbox, entity.scale):
                LodMapCreator.appendFrontPlaneIndicesForLod(diffuseSamplerToIndices[diffuseSampler], len(diffuseSamplerToVertices[diffuseSampler]))
                self.appendDiagonalPlaneVerticesForLod(diffuseSamplerToVertices[diffuseSampler], diffuseSamplerToNormals[diffuseSampler], diffuseSamplerToTextureUVs[diffuseSampler], entity, planeIntersection)

            if lodCandidate.hasTop(bbox, entity.scale):
                LodMapCreator.appendTopPlaneIndicesForLod(diffuseSamplerToIndices[diffuseSampler], len(diffuseSamplerToVertices[diffuseSampler]))
                self.appendTopPlaneVerticesForLod(diffuseSamplerToVertices[diffuseSampler], diffuseSamplerToNormals[diffuseSampler], diffuseSamplerToTextureUVs[diffuseSampler], entity, planeIntersection)

        totalBoundingGeometry = BoundingGeometry()
        for diffuseSampler in diffuseSamplerToVertices:
            totalBoundingGeometry.extendByPoints(diffuseSamplerToVertices[diffuseSampler])

        totalBoundingSphere = totalBoundingGeometry.getSphere()
        center = totalBoundingSphere.center
        translation = np.multiply(center, [-1]).tolist()

        totalBoundingBox = totalBoundingGeometry.getBox().getTranslated(translation)
        totalBoundingSphere = totalBoundingSphere.getTranslated(translation)

        bounds = self.createAabb(totalBoundingBox)

        shaders = ""
        geometries = ""
        shaderIndex = 0
        for diffuseSampler in diffuseSamplerToVertices:
            indices = diffuseSamplerToIndices[diffuseSampler]

            boundingGeometry = BoundingGeometry(diffuseSamplerToVertices[diffuseSampler])
            boundingBox = boundingGeometry.getBox().getTranslated(translation)

            verticesNormalsTextureUVsStr = LodMapCreator.convertVerticesNormalsTextureUVsAsStr(diffuseSamplerToVertices[diffuseSampler], diffuseSamplerToNormals[diffuseSampler], diffuseSamplerToTextureUVs[diffuseSampler], translation)

            shaders += self.contentTemplateOdrShaderTreeLod.replace("${DIFFUSE_SAMPLER}", diffuseSampler)

            bounds += self.createAabb(boundingBox)

            geometries += self.contentTemplateMeshGeometry \
                .replace("${SHADER_INDEX}", str(shaderIndex)) \
                .replace("${VERTEX_DECLARATION}", "N209731BE") \
                .replace("${INDICES.NUM}", str(len(indices))) \
                .replace("${INDICES}", self.createIndicesStr(indices)) \
                .replace("${VERTICES.NUM}", str(len(diffuseSamplerToVertices[diffuseSampler]))) \
                .replace("${VERTICES}\n", verticesNormalsTextureUVsStr)
            shaderIndex += 1

        contentModelMesh = self.contentTemplateMesh \
            .replace("${BOUNDS}\n", bounds) \
            .replace("${GEOMETRIES}\n", geometries)

        fileModelMesh = open(os.path.join(self.getOutputDirMeshes(), lodName.lower() + ".mesh"), 'w')
        fileModelMesh.write(contentModelMesh)
        fileModelMesh.close()

        contentModelOdr = self.contentTemplateOdr \
            .replace("${BBOX.MIN.X}", Util.floatToStr(totalBoundingBox.min[0])) \
            .replace("${BBOX.MIN.Y}", Util.floatToStr(totalBoundingBox.min[1])) \
            .replace("${BBOX.MIN.Z}", Util.floatToStr(totalBoundingBox.min[2])) \
            .replace("${BBOX.MAX.X}", Util.floatToStr(totalBoundingBox.max[0])) \
            .replace("${BBOX.MAX.Y}", Util.floatToStr(totalBoundingBox.max[1])) \
            .replace("${BBOX.MAX.Z}", Util.floatToStr(totalBoundingBox.max[2])) \
            .replace("${BSPHERE.CENTER.X}", Util.floatToStr(totalBoundingSphere.center[0])) \
            .replace("${BSPHERE.CENTER.Y}", Util.floatToStr(totalBoundingSphere.center[1])) \
            .replace("${BSPHERE.CENTER.Z}", Util.floatToStr(totalBoundingSphere.center[2])) \
            .replace("${BSPHERE.RADIUS}", Util.floatToStr(totalBoundingSphere.radius)) \
            .replace("${MESH_FILENAME}", lodName.lower() + ".mesh") \
            .replace("${SHADERS}\n", shaders)

        fileModelOdr = open(os.path.join(self.getOutputDirMeshes(), lodName.lower() + ".odr"), 'w')
        fileModelOdr.write(contentModelOdr)
        fileModelOdr.close()

        itemHdDistance = 300
        itemLodDistance = LodMapCreator.LOD_DISTANCE

        if not os.path.exists(os.path.join(self.getOutputDirModels(), self.getYtypName() + ".ytyp.xml")):
            self.slodYtypItems = open(os.path.join(self.getOutputDirModels(), self.getYtypName() + ".ytyp.xml"), 'w')
            self.slodYtypItems.write("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
        <CMapTypes>
          <extensions/>
          <archetypes>
        """)
        self.slodYtypItems.write(self.replacePlaceholders(self.contentTemplateYtypItem, lodName, self.prefix + "_lod", drawableDictionary, totalBoundingBox, totalBoundingSphere, itemHdDistance, itemLodDistance))

        return EntityItem(lodName, center, [1, 1, 1], [1, 0, 0, 0], itemLodDistance, itemHdDistance, parentIndex, numChildren, LodLevel.LOD, Flag.FLAGS_LOD)

    def calculateVectorOnEllipseAtDiagonal(self, semiaxisX: float, semiaxisY: float, quadrant: int) -> list[float]:
        coordinate = semiaxisX * semiaxisY / math.sqrt(semiaxisX**2 + semiaxisY**2)
        return [coordinate * (1 if quadrant == 0 or quadrant == 3 else -1), coordinate * (1 if quadrant == 0 or quadrant == 1 else -1)]

    def createDrawableDictionary(self, name: str, entities: list[EntityItem]):
        if len(entities) == 0:
            return

        relPath = os.path.relpath(self.getOutputDirMeshes(), self.getOutputDirModels())

        file = open(os.path.join(self.getOutputDirModels(), name.lower() + ".odd"), 'w')
        file.write("Version 165 32\n{\n")
        for entity in entities:
            file.write("\t")
            file.write(os.path.join(relPath, entity.archetypeName.lower() + ".odr"))
            file.write("\n")
        file.write("}\n")
        file.close()

    def appendSlodTop(self, verticesTop: list[list[float]], normalsTop: list[list[float]], textureUVsTop: list[list[float]], size: list[float], center: list[float], rotation: list[float], uvMap: UVMap):
        # in terms of absolute z-coordinate this is sizeZ * (1 - topZ) + transformedBboxEntityMin[2]
        # which yields relative (to centerTransformed) z-coordinate sizeZ * (1 - topZ) - sizeZ / 2 <=>  sizeZ * ((1 - topZ) - 1 / 2)
        planeZ = size[1] * (0.5 - uvMap.topZ)

        vertices = [
            [-size[0] / 2, -size[0] / 2, planeZ],
            [size[0] / 2, -size[0] / 2, planeZ],
            [size[0] / 2, size[0] / 2, planeZ],
            [-size[0] / 2, size[0] / 2, planeZ]
        ]

        normals = [
            [-1, -1, 0.1],
            [1, -1, 0.1],
            [1, 1, 0.1],
            [-1, 1, 0.1]
        ]

        rotZ, unused, unused = transforms3d.euler.quat2euler(rotation, axes='rzyx')
        onlyZRotationQuaternion = transforms3d.euler.euler2quat(rotZ, 0, 0, axes='rzyx')
        for i in range(4):
            rotatedVertex = Util.applyRotation(vertices[i], onlyZRotationQuaternion)
            translatedVertex = np.add(rotatedVertex, center).tolist()
            rotatedNormal = Util.applyRotation(normals[i], onlyZRotationQuaternion)

            verticesTop.append(translatedVertex)
            normalsTop.append(rotatedNormal)

        textureUVsTop += [
            [uvMap.topMin.u, uvMap.topMax.v],
            [uvMap.topMax.u, uvMap.topMax.v],
            [uvMap.topMax.u, uvMap.topMin.v],
            [uvMap.topMin.u, uvMap.topMin.v]
        ]

    def createSlodModel(self, nameWithoutSlodLevel: str, slodLevel: int, drawableDictionary: str, entities: list[EntityItem], parentIndex: int, numChildren: int, lodLevel: str, flags: int) -> EntityItem:
        name = nameWithoutSlodLevel
        if slodLevel > 0:
            name += str(slodLevel)

        verticesFront = {}
        sizesFront = {}
        textureUVsFront = {}
        verticesTop = {}
        normalsTop = {}
        textureUVsTop = {}
        bbox = {}
        for entity in entities:
            uvMap = self.slodCandidates[entity.archetypeName]

            diffuseSampler = uvMap.getDiffuseSampler()

            if diffuseSampler not in verticesFront:
                verticesFront[diffuseSampler] = []
                sizesFront[diffuseSampler] = []
                textureUVsFront[diffuseSampler] = []
                verticesTop[diffuseSampler] = []
                normalsTop[diffuseSampler] = []
                textureUVsTop[diffuseSampler] = []
                bbox[diffuseSampler] = Box.createReversedInfinityBox()

            bboxEntity = self.ytypItems[entity.archetypeName].boundingBox

            size = bboxEntity.getScaled(entity.scale).getSizes()
            sizeXY = (size[0] + size[1]) / 2
            sizeZ = size[2]

            centerTransformed = entity.applyTransformationTo(bboxEntity.getCenter())

            transformedBboxEntityMin = np.subtract(centerTransformed, [sizeXY / 2, sizeXY / 2, sizeZ / 2]).tolist()
            transformedBboxEntityMax = np.add(centerTransformed, [sizeXY / 2, sizeXY / 2, sizeZ / 2]).tolist()

            bbox[diffuseSampler].extendByPoint(transformedBboxEntityMin)
            bbox[diffuseSampler].extendByPoint(transformedBboxEntityMax)

            size2D = [sizeXY, sizeZ]
            sizesFront[diffuseSampler].append(size2D)
            verticesFront[diffuseSampler].append(centerTransformed)
            textureUVsFront[diffuseSampler].append([uvMap.frontMin, uvMap.frontMax])

            if slodLevel < 3 and uvMap.topMin is not None and uvMap.topMax is not None:
                assert uvMap.topZ is not None
                self.appendSlodTop(verticesTop[diffuseSampler], normalsTop[diffuseSampler], textureUVsTop[diffuseSampler], size2D, centerTransformed, entity.rotation, uvMap)

        totalBoundingGeometry = BoundingGeometry()

        for diffuseSampler in verticesTop:
            totalBoundingGeometry.extendByPoints(verticesTop[diffuseSampler])

        for diffuseSampler in verticesFront:
            for i in range(len(verticesFront[diffuseSampler])):
                size2D = sizesFront[diffuseSampler][i]
                size3D = [size2D[0], size2D[0], size2D[1]]
                minVertex = np.subtract(verticesFront[diffuseSampler][i], size3D).tolist()
                maxVertex = np.add(verticesFront[diffuseSampler][i], size3D).tolist()
                totalBoundingGeometry.extendByPoints([minVertex, maxVertex])

        totalBoundingSphere = totalBoundingGeometry.getSphere()
        center = totalBoundingSphere.center
        translation = np.multiply(center, [-1]).tolist()

        totalBoundingBox = totalBoundingGeometry.getBox().getTranslated(translation)
        totalBoundingSphere = totalBoundingSphere.getTranslated(translation)

        bounds = self.createAabb(totalBoundingBox)

        shaders = ""
        geometries = ""
        shaderIndex = 0
        for diffuseSampler in verticesFront:
            numFrontPlanes = len(verticesFront[diffuseSampler])
            indicesFront = self.createIndicesForRectangles(numFrontPlanes)

            bbox[diffuseSampler] = bbox[diffuseSampler].getTranslated(translation)
            verticesFrontStr = LodMapCreator.convertVerticesTextureUVsAsStrForSlod(verticesFront[diffuseSampler], sizesFront[diffuseSampler], textureUVsFront[diffuseSampler], translation)

            shaders += self.contentTemplateOdrShaderTreeLod2.replace("${DIFFUSE_SAMPLER}", diffuseSampler)

            bounds += self.createAabb(bbox[diffuseSampler])

            geometries += self.contentTemplateMeshGeometry \
                .replace("${SHADER_INDEX}", str(shaderIndex)) \
                .replace("${VERTEX_DECLARATION}", "N5A9A1E1A") \
                .replace("${INDICES.NUM}", str(numFrontPlanes * 6)) \
                .replace("${INDICES}", self.createIndicesStr(indicesFront)) \
                .replace("${VERTICES.NUM}", str(numFrontPlanes * 4)) \
                .replace("${VERTICES}\n", verticesFrontStr)
            shaderIndex += 1

            if len(verticesTop[diffuseSampler]) > 0:
                assert len(verticesTop[diffuseSampler]) % 4 == 0
                numTopPlanes = int(len(verticesTop[diffuseSampler]) / 4)
                indicesTop = self.createIndicesForRectangles(numTopPlanes)
                verticesTopStr = LodMapCreator.convertVerticesNormalsTextureUVsAsStr(verticesTop[diffuseSampler], normalsTop[diffuseSampler], textureUVsTop[diffuseSampler], translation)

                shaders += self.contentTemplateOdrShaderTreeLod.replace("${DIFFUSE_SAMPLER}", diffuseSampler + "_top")

                bounds += self.createAabb(bbox[diffuseSampler])

                geometries += self.contentTemplateMeshGeometry \
                    .replace("${SHADER_INDEX}", str(shaderIndex)) \
                    .replace("${VERTEX_DECLARATION}", "N209731BE") \
                    .replace("${INDICES.NUM}", str(numTopPlanes * 6)) \
                    .replace("${INDICES}", self.createIndicesStr(indicesTop)) \
                    .replace("${VERTICES.NUM}", str(len(verticesTop[diffuseSampler]))) \
                    .replace("${VERTICES}\n", verticesTopStr)
                shaderIndex += 1

        contentModelMesh = self.contentTemplateMesh \
            .replace("${BOUNDS}\n", bounds) \
            .replace("${GEOMETRIES}\n", geometries)

        fileModelMesh = open(os.path.join(self.getOutputDirMeshes(), name.lower() + ".mesh"), 'w')
        fileModelMesh.write(contentModelMesh)
        fileModelMesh.close()

        contentModelOdr = self.contentTemplateOdr \
            .replace("${BBOX.MIN.X}", Util.floatToStr(totalBoundingBox.min[0])) \
            .replace("${BBOX.MIN.Y}", Util.floatToStr(totalBoundingBox.min[1])) \
            .replace("${BBOX.MIN.Z}", Util.floatToStr(totalBoundingBox.min[2])) \
            .replace("${BBOX.MAX.X}", Util.floatToStr(totalBoundingBox.max[0])) \
            .replace("${BBOX.MAX.Y}", Util.floatToStr(totalBoundingBox.max[1])) \
            .replace("${BBOX.MAX.Z}", Util.floatToStr(totalBoundingBox.max[2])) \
            .replace("${BSPHERE.CENTER.X}", Util.floatToStr(totalBoundingSphere.center[0])) \
            .replace("${BSPHERE.CENTER.Y}", Util.floatToStr(totalBoundingSphere.center[1])) \
            .replace("${BSPHERE.CENTER.Z}", Util.floatToStr(totalBoundingSphere.center[2])) \
            .replace("${BSPHERE.RADIUS}", Util.floatToStr(totalBoundingSphere.radius)) \
            .replace("${MESH_FILENAME}", name.lower() + ".mesh") \
            .replace("${SHADERS}\n", shaders)

        fileModelOdr = open(os.path.join(self.getOutputDirMeshes(), name.lower() + ".odr"), 'w')
        fileModelOdr.write(contentModelOdr)
        fileModelOdr.close()

        if slodLevel == 0:
            itemHdDistance = 350
            itemLodDistance = LodMapCreator.LOD_DISTANCE
        elif slodLevel == 1:
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
        self.slodYtypItems.write(self.replacePlaceholders(self.contentTemplateYtypItem, name, self.getYtypName(), drawableDictionary, totalBoundingBox, totalBoundingSphere, itemHdDistance, itemLodDistance))

        return EntityItem(name, center, [1, 1, 1], [1, 0, 0, 0], itemLodDistance, itemHdDistance, parentIndex, numChildren, lodLevel, flags)

    def fixHdOrOrphanHdLodLevelsAndRearrangeEntites(self, content: str) -> str:
        hdEntities = ""
        orphanHdEntites = ""
        for match in re.finditer('(\\s*<Item type="CEntityDef">' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*<flags value=")([^"]+)("\\s*/>' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*<parentIndex value="([^"]+)"\\s*/>' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*<lodLevel>)[^<]+(</lodLevel>' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*<priorityLevel>)([^<]+)(</priorityLevel>' +
                                 '(?:\\s*<[^/].*>)*?' +
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

        matchEntities = re.search("<entities>", content)
        if matchEntities is None:
            return content

        start = matchEntities.end()
        end = re.search("\\s+</entities>[\\S\\s]*?\\Z", content, re.M).start()

        return content[:start] + hdEntities + orphanHdEntites + content[end:]

    def resetParentIndexAndNumChildren(self, content: str) -> str:
        result = re.sub('(<parentIndex value=")[^"]+("/>)', '\\g<1>-1\\g<2>', content)
        result = re.sub('(<numChildren value=")[^"]+("/>)', '\\g<1>0\\g<2>', result)
        return result

    def determinePrefixBundles(self):
        mapNames = []
        for filename in natsorted(os.listdir(self.inputDir)):
            if filename.endswith(".ymap.xml") and not filename.endswith("_lod.ymap.xml") and not filename.endswith("_slod2.ymap.xml"):
                mapNames.append(filename[:-9])

        self.bundlePrefixes = Util.determinePrefixBundles(mapNames)

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
        lodCandidateNames = list(self.lodCandidates.keys())

        hdEntitiesWithLod = []
        lodCoords = []
        lodDistances = []
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml") or not (filename.startswith(mapPrefix.lower() + "_") or filename == mapPrefix.lower() + ".ymap.xml"):
                continue

            mapName = filename[:-9]

            if os.path.exists(os.path.join(self.getOutputDirMaps(), mapName + ".ymap.xml")):
                print("\twarning: skipping " + filename + " since such a map was created by this script")
                continue

            print("\tprocessing " + filename)

            fileNoLod = open(os.path.join(self.inputDir, filename), 'r')
            contentNoLod = fileNoLod.read()
            fileNoLod.close()

            contentNoLod = self.resetParentIndexAndNumChildren(contentNoLod)
            contentNoLod = Ymap.replaceName(contentNoLod, mapName)
            contentNoLod = Ymap.calculateAndReplaceLodDistance(contentNoLod, self.ytypItems, archetypes=lodCandidateNames, forceHasParent=True)

            fileNoLod = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
            fileNoLod.write(contentNoLod)
            fileNoLod.close()

            pattern = re.compile('[\t ]*<Item type="CEntityDef">' +
                                 '\\s*<archetypeName>([^<]+)</archetypeName>' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*<position x="([^"]+)" y="([^"]+)" z="([^"]+)"\\s*/>' +
                                 '\\s*<rotation x="([^"]+)" y="([^"]+)" z="([^"]+)" w="([^"]+)"\\s*/>' +
                                 '\\s*<scaleXY value="([^"]+)"\\s*/>' +
                                 '\\s*<scaleZ value="([^"]+)"\\s*/>' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*<lodDist value="([^"]+)"\\s*/>' +
                                 '(?:\\s*<[^/].*>)*?' +
                                 '\\s*</Item>[\r\n]+')

            for matchobj in re.finditer(pattern, contentNoLod):
                archetypeName = matchobj.group(1).lower()
                if (not self.USE_SLOD_AS_LOD_MODEL and archetypeName not in self.lodCandidates) or \
                        (self.USE_SLOD_AS_LOD_MODEL and archetypeName not in self.slodCandidates):
                    continue

                position = [float(matchobj.group(2)), float(matchobj.group(3)), float(matchobj.group(4))]
                rotation = [float(matchobj.group(8)), -float(matchobj.group(5)), -float(matchobj.group(6)), -float(matchobj.group(7))]  # order is w, -x, -y, -z
                scale = [float(matchobj.group(9)), float(matchobj.group(9)), float(matchobj.group(10))]
                lodDistance = float(matchobj.group(11))
                entity = EntityItem(archetypeName, position, scale, rotation, lodDistance)

                hdEntitiesWithLod.append(entity)

                lodCoords.append(position)
                lodDistances.append(lodDistance)

        hierarchy = self.calculateLodHierarchy(lodCoords, lodDistances)

        entitiesForLodModels = {}
        entitiesForSlod1Models = {}
        entitiesForSlod2Models = {}
        entitiesForSlod3Models = {}

        hdToLod = {}
        lodToSlod1 = {}
        slod1ToSlod2 = {}
        slod2ToSlod3 = {}

        lodNumChildren = {}
        slod1NumChildren = {}
        slod2NumChildren = {}
        slod3NumChildren = {}

        for i in range(len(hdEntitiesWithLod)):
            hdEntity = hdEntitiesWithLod[i]
            h = hierarchy[i]

            hdToLod[i] = h[0]
            lodNumChildren[h[0]] = lodNumChildren.get(h[0], 0) + 1
            if h[0] not in entitiesForLodModels:
                entitiesForLodModels[h[0]] = []
            entitiesForLodModels[h[0]].append(hdEntity)

            if hdEntity.archetypeName in self.slodCandidates:
                if hdEntity.lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD1:
                    if h[0] not in lodToSlod1:
                        lodToSlod1[h[0]] = h[2]
                        slod1NumChildren[h[2]] = slod1NumChildren.get(h[2], 0) + 1

                    if h[2] not in entitiesForSlod1Models:
                        entitiesForSlod1Models[h[2]] = []
                    entitiesForSlod1Models[h[2]].append(hdEntity)
                if hdEntity.lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD2:
                    if h[2] not in slod1ToSlod2:
                        slod1ToSlod2[h[2]] = h[3]
                        slod2NumChildren[h[3]] = slod2NumChildren.get(h[3], 0) + 1

                    if h[3] not in entitiesForSlod2Models:
                        entitiesForSlod2Models[h[3]] = []
                    entitiesForSlod2Models[h[3]].append(hdEntity)
                if hdEntity.lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD3:
                    if h[3] not in slod2ToSlod3:
                        slod2ToSlod3[h[3]] = h[4]
                        slod3NumChildren[h[4]] = slod3NumChildren.get(h[4], 0) + 1

                    if h[4] not in entitiesForSlod3Models:
                        entitiesForSlod3Models[h[4]] = []
                    entitiesForSlod3Models[h[4]].append(hdEntity)

        lodDrawableDictionary = mapPrefix.lower() + "_lod_children"
        slod1DrawableDictionary = mapPrefix.lower() + "_slod1_children"
        slod2DrawableDictionary = mapPrefix.lower() + "_slod2_children"

        lodEntities = [[]]
        slod1Entities = [[]]
        slod2Entities = []
        slod3Entities = []

        index = 0
        slod3KeyToIndex = {}
        for key in sorted(entitiesForSlod3Models):
            slodName = mapPrefix.lower() + "_" + str(index) + "_slod"
            slod3Entities.append(self.createSlodModel(
                slodName, 3, slod2DrawableDictionary,
                entitiesForSlod3Models[key],
                -1, slod3NumChildren[key],
                LodLevel.SLOD3, Flag.FLAGS_SLOD3
            ))
            slod3KeyToIndex[key] = index
            index += 1

        index = 0
        slod2KeyToIndex = {}
        for key in sorted(entitiesForSlod2Models):
            slodName = mapPrefix.lower() + "_" + str(index) + "_slod"
            parentIndex = self.getParentIndexForKey(key, slod2ToSlod3, slod3KeyToIndex, 0)
            slod2Entities.append(self.createSlodModel(
                slodName, 2, slod2DrawableDictionary,
                entitiesForSlod2Models[key],
                parentIndex, slod2NumChildren[key],
                LodLevel.SLOD2, Flag.FLAGS_SLOD2
            ))
            slod2KeyToIndex[key] = index
            index += 1

        index = 0
        slod1KeyToIndex = {}
        parentIndexOffset = len(slod3Entities)
        for key in sorted(entitiesForSlod1Models):
            slodName = mapPrefix.lower() + "_" + str(index) + "_slod"
            parentIndex = self.getParentIndexForKey(key, slod1ToSlod2, slod2KeyToIndex, parentIndexOffset)

            if len(slod1Entities[-1]) >= LodMapCreator.MAX_NUM_CHILDREN_IN_DRAWABLE_DICTIONARY:
                slod1Entities.append([])

            slod1Entities[-1].append(self.createSlodModel(
                slodName, 1, slod1DrawableDictionary + "_" + str(len(slod1Entities) - 1),
                entitiesForSlod1Models[key],
                parentIndex, slod1NumChildren[key],
                LodLevel.SLOD1, Flag.FLAGS_SLOD1
            ))
            slod1KeyToIndex[key] = index
            index += 1

        for key in sorted(entitiesForLodModels):
            lodName = mapPrefix.lower() + "_" + str(key) + "_lod"
            parentIndex = self.getParentIndexForKey(key, lodToSlod1, slod1KeyToIndex, 0)

            if len(lodEntities[-1]) >= LodMapCreator.MAX_NUM_CHILDREN_IN_DRAWABLE_DICTIONARY:
                lodEntities.append([])

            if self.USE_SLOD_AS_LOD_MODEL:
                lodEntities[-1].append(self.createSlodModel(
                    lodName, 0, lodDrawableDictionary + "_" + str(len(lodEntities) - 1),
                    entitiesForLodModels[key],
                    parentIndex, lodNumChildren[key],
                    LodLevel.LOD, Flag.FLAGS_LOD
                ))
            else:
                lodEntities[-1].append(self.createLodModel(
                    lodName,
                    lodDrawableDictionary + "_" + str(len(lodEntities) - 1),
                    entitiesForLodModels[key],
                    parentIndex, lodNumChildren[key]
                ))

        for lodEntitiesIndex in range(len(lodEntities)):
            self.createDrawableDictionary(lodDrawableDictionary + "_" + str(lodEntitiesIndex), lodEntities[lodEntitiesIndex])
        for slod1EntitiesIndex in range(len(slod1Entities)):
            self.createDrawableDictionary(slod1DrawableDictionary + "_" + str(slod1EntitiesIndex), slod1Entities[slod1EntitiesIndex])
        self.createDrawableDictionary(slod2DrawableDictionary, slod2Entities + slod3Entities)

        if len(slod2Entities) == 0:
            slod2MapName = None
        else:
            slod2MapName = mapPrefix.lower() + "_slod2"
            self.writeLodOrSlodMap(slod2MapName, None, ContentFlag.SLOD + ContentFlag.SLOD2, slod3Entities + slod2Entities)

        slod1AndLodEntities = []
        numSlod1Entities = 0
        for slod1s in slod1Entities:
            slod1AndLodEntities += slod1s
            numSlod1Entities += len(slod1s)
        for lods in lodEntities:
            slod1AndLodEntities += lods
        if len(slod1AndLodEntities) > 0:
            self.foundLod = True
            if numSlod1Entities > 0:
                self.foundSlod = True
            self.writeLodOrSlodMap(mapPrefix.lower() + "_lod", slod2MapName, ContentFlag.LOD + ContentFlag.SLOD, slod1AndLodEntities)

        self.adaptHdMapsForPrefix(mapPrefix, hdToLod, numSlod1Entities)

    def getParentIndexForKey(self, key: int, keyToParentKey: dict[int, int], parentKeyToIndex: dict[int, int], parentIndexOffset: int):
        if key not in keyToParentKey:
            return -1
        else:
            return parentKeyToIndex[keyToParentKey[key]] + parentIndexOffset

    def adaptHdMapsForPrefix(self, mapPrefix: str, hdToLod: dict[int, int], offsetParentIndex: int):
        mutableIndex = [0]
        for filename in natsorted(os.listdir(self.inputDir)):
            if not filename.endswith(".ymap.xml") or not (filename.startswith(mapPrefix.lower() + "_") or filename == mapPrefix.lower() + ".ymap.xml"):
                continue

            fileNoLod = open(os.path.join(self.getOutputDirMaps(), filename), 'r')
            contentNoLod = fileNoLod.read()
            fileNoLod.close()

            indexBefore = mutableIndex[0]
            # fix parentIndex in hd map to match lod map
            contentNoLod = re.sub('(\\s*<Item type="CEntityDef">' +
                                  '\\s*<archetypeName>([^<]+)</archetypeName>' +
                                  '(?:\\s*<[^/].*>)*?' +
                                  '\\s*<parentIndex value=")[^"]+("\\s*/>' +
                                  '(?:\\s*<[^/].*>)*?' +
                                  '\\s*</Item>)', lambda match: self.replParentIndex(match, mutableIndex, hdToLod, offsetParentIndex), contentNoLod, flags=re.M)

            contentNoLod = Ymap.replaceParent(contentNoLod, None if indexBefore == mutableIndex[0] else mapPrefix.lower() + "_lod")
            contentNoLod = self.fixHdOrOrphanHdLodLevelsAndRearrangeEntites(contentNoLod)

            fileNoLod = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
            fileNoLod.write(contentNoLod)
            fileNoLod.close()

    def writeLodOrSlodMap(self, mapName: str, parentMap: Optional[str], contentFlags: int, entities: list[EntityItem]):
        contentEntities = self.createEntitiesContent(entities)

        content = self.contentTemplateSlod2Map \
            .replace("${NAME}", mapName) \
            .replace("${CONTENT_FLAGS}", str(contentFlags)) \
            .replace("${TIMESTAMP}", Util.getNowInIsoFormat()) \
            .replace("${ENTITIES}\n", contentEntities)

        content = Ymap.replaceParent(content, parentMap)

        fileMap = open(os.path.join(self.getOutputDirMaps(), mapName + ".ymap.xml"), 'w')
        fileMap.write(content)
        fileMap.close()

    def createEntitiesContent(self, entities: list[EntityItem]):
        contentEntities = ""
        for entity in entities:
            contentEntities += self.contentTemplateEntitySlod \
                .replace("${POSITION.X}", Util.floatToStr(entity.position[0])) \
                .replace("${POSITION.Y}", Util.floatToStr(entity.position[1])) \
                .replace("${POSITION.Z}", Util.floatToStr(entity.position[2])) \
                .replace("${NAME}", entity.archetypeName) \
                .replace("${NUM_CHILDREN}", str(entity.numChildren)) \
                .replace("${PARENT_INDEX}", str(entity.parentIndex)) \
                .replace("${LOD_LEVEL}", entity.lodLevel) \
                .replace("${CHILD.LOD_DISTANCE}", Util.floatToStr(entity.childLodDist)) \
                .replace("${LOD_DISTANCE}", Util.floatToStr(entity.lodDistance)) \
                .replace("${FLAGS}", str(entity.flags))
        return contentEntities

    def calculateLodHierarchy(self, points: list[list[float]], lodDistances: list[float]) -> list[list[int]]:
        if len(points) == 0:
            return []

        hierarchy = []
        for i in range(len(points)):
            hierarchy.append([])

        return self._calculateLodHierarchy(points, lodDistances, hierarchy)

    def _calculateLodHierarchy(self, points: list[list[float]], lodDistances: list[float], hierarchy: list[list[int]]) -> list[list[int]]:
        numMaxChildren = -1
        clusterWithRespectToPosition = True
        level = len(hierarchy[0])
        if level == 0:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD3
        elif level == 1:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD2
        elif level == 2:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD1
        elif level == 3:
            maxExtends = LodMapCreator.LOD_DISTANCES_MAX_DIFFERENCE_LOD
            clusterWithRespectToPosition = False
        elif level == 4:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_LOD
            numMaxChildren = LodMapCreator.NUM_CHILDREN_MAX_VALUE
        else:
            return hierarchy

        absIndices = []
        pointsOfParent = []
        for i in range(len(points)):
            parentIndex = 0 if len(hierarchy[i]) == 0 else hierarchy[i][0]

            while parentIndex >= len(pointsOfParent):
                absIndices.append([])
                pointsOfParent.append([])

            absIndices[parentIndex].append(i)
            pointsOfParent[parentIndex].append(points[i] if clusterWithRespectToPosition else lodDistances[i])

        if not clusterWithRespectToPosition:
            for i in range(len(points)):
                parentIndex = 0 if len(hierarchy[i]) == 0 else hierarchy[i][0]
                pointsOfParent[parentIndex] = np.array(pointsOfParent[parentIndex]).reshape(-1, 1)

        clusterOffset = 0
        for parentIndex in range(len(pointsOfParent)):
            clustering, unused = Util.performClustering(pointsOfParent[parentIndex], numMaxChildren, maxExtends, not clusterWithRespectToPosition)

            for c in range(len(clustering)):
                i = absIndices[parentIndex][c]
                hierarchy[i].insert(0, clustering[c] + clusterOffset)

            clusterOffset += max(clustering) + 1

        return self._calculateLodHierarchy(points, lodDistances, hierarchy)

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.getOutputDirMaps(),
            lambda filename: not filename.endswith(".ymap.xml") or not filename.startswith(tuple(each.lower() for each in self.bundlePrefixes)))

    # adapt extents and set current datetime
    def fixMapExtents(self):
        print("\tfixing map extents")

        for filename in natsorted(os.listdir(self.getOutputDirMaps())):
            if not filename.endswith(".ymap.xml"):
                continue

            file = open(os.path.join(self.getOutputDirMaps(), filename), 'r')
            content = file.read()
            file.close()

            content = Ymap.fixMapExtents(content, self.ytypItems)

            file = open(os.path.join(self.getOutputDirMaps(), filename), 'w')
            file.write(content)
            file.close()

    def createManifest(self) -> None:
        print("\tcreating manifest")
        manifest = Manifest(self.ytypItems, self.getOutputDirMaps())
        manifest.parseYmaps()
        manifest.writeManifest()

    def addLodAndSlodModelsToYtypDict(self) -> None:
        if self.slodYtypItems is not None:
            self.ytypItems |= YtypParser.readYtypDirectory(self.getOutputDirModels())

    def copyTextureDictionaries(self):
        texturesDir = os.path.join(os.path.dirname(__file__), "textures")
        if self.foundLod:
            shutil.copyfile(os.path.join(texturesDir, "lod.ytd"), os.path.join(self.getOutputDirModels(), self.prefix + "_lod.ytd"))
        if self.foundSlod:
            shutil.copyfile(os.path.join(texturesDir, "slod.ytd"), os.path.join(self.getOutputDirModels(), self.prefix + "_slod.ytd"))
