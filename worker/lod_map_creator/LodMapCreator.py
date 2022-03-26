import shutil
from re import Match
from typing import Any, IO, Optional

import numpy as np
import transforms3d
from natsort import natsorted

import os
import re

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

    lodCandidates: dict[str, LodCandidate]

    def prepareLodCandidates(self):
        lodCandidates = {
            # trees
            "prop_tree_birch_01": LodCandidate("prop_tree_birch_01", 0.546875, 0.6640625, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1),
                UV(0.7734375, 0.5078125)),
            "prop_tree_birch_02": LodCandidate("prop_tree_birch_02", 0.421875, 0.5703125, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1),
                UV(0.765625, 0.3671875)),
            "prop_tree_birch_03": LodCandidate("prop_tree_birch_03", 0.546875),
            "prop_tree_birch_03b": LodCandidate("prop_tree_birch_03b", 0.5625),
            "prop_tree_birch_04": LodCandidate("prop_tree_birch_04", 0.5625, 0.421875, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0),
                UV(0.7734375, 0.453125)),
            "prop_tree_maple_02": LodCandidate("prop_tree_maple_02", 0.421875),
            # TODO provide top view
            "prop_tree_maple_03": LodCandidate("prop_tree_maple_03", 0.5),
            "prop_tree_cedar_02": LodCandidate("prop_tree_cedar_02", 0.515625, 0.46315789473, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_cedar_03": LodCandidate("prop_tree_cedar_03", 0.5390625, 0.51052631578, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_tree_cedar_04": LodCandidate("prop_tree_cedar_04", 0.484375, 0.3947368421, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1),
                UV(0.484375, 0.87890625)),
            "prop_tree_cedar_s_01": LodCandidate("prop_tree_cedar_s_01", 0.484375, 0.71875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1),
                UV(0.46875, 0.8203125)),
            "prop_tree_cedar_s_04": LodCandidate("prop_tree_cedar_s_04", 0.5, 0.68269230769, UV(0, 0), UV(1, 0.8125), UV(0, 1), UV(1, 0.8125)),
            "prop_tree_cedar_s_05": LodCandidate("prop_tree_cedar_s_05", 0.46875),
            "prop_tree_cedar_s_06": LodCandidate("prop_tree_cedar_s_06", 0.5),
            "prop_tree_cypress_01": LodCandidate("prop_tree_cypress_01", 0.5),
            "prop_tree_eng_oak_01": LodCandidate("prop_tree_eng_oak_01", 0.5, 0.5703125, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1), UV(0.75, 0.53125)),
            "prop_tree_eucalip_01": LodCandidate("prop_tree_eucalip_01", 0.5, 0.359375, UV(0, 0), UV(0.5, 1), UV(0.5, 0), UV(1, 1)),
            "prop_tree_jacada_01": LodCandidate("prop_tree_jacada_01", 0.484375, 0.46875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_jacada_02": LodCandidate("prop_tree_jacada_02", 0.515625, 0.546875, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_oak_01": LodCandidate("prop_tree_oak_01", 0.4765625, 0.4765625, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1), UV(0.46875, 0.75)),
            "prop_tree_olive_01": LodCandidate("prop_tree_olive_01", 0.5, 0.40625, UV(0, 0), UV(1, 0.5), UV(0, 1), UV(1, 0.5)),
            "prop_tree_pine_01": LodCandidate("prop_tree_pine_01", 0.515625, 0.515625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.515625, 0.79296875)),
            "prop_tree_pine_02": LodCandidate("prop_tree_pine_02", 0.546875, 0.6875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.5, 0.80078125)),
            "prop_tree_fallen_pine_01": LodCandidate("prop_tree_fallen_pine_01", 0.609375, 1, UV(0, 0), UV(1, 0.625)),
            # , UV(0, 1), UV(1, 0.625), UV(0.390625, 0.7734375)),
            "prop_s_pine_dead_01": LodCandidate("prop_s_pine_dead_01", 0.40625, 0.4875, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.53125, 0.8515625)),
            "prop_w_r_cedar_01": LodCandidate("prop_w_r_cedar_01", 0.5, 0.8, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_w_r_cedar_dead": LodCandidate("prop_w_r_cedar_dead", 0.59375, 0.425, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.53125, 0.78125)),
            "test_tree_cedar_trunk_001": LodCandidate("test_tree_cedar_trunk_001", 0.515625, 0.5769231, UV(0, 0), UV(1, 0.8125), UV(0, 1),
                UV(1, 0.8125)),
            "test_tree_forest_trunk_01": LodCandidate("test_tree_forest_trunk_01", 0.515625, 0.3894231, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125),
                UV(1, 1)),
            "test_tree_forest_trunk_04": LodCandidate("test_tree_forest_trunk_04", 0.453125, 1, UV(0, 0), UV(0.5, 1), UV(0.5, 1), UV(1, 0),
                UV(0.78125, 0.5546875), 0.6484375),
            # trees2
            "prop_tree_lficus_02": LodCandidate("prop_tree_lficus_02", 0.4453125, 0.55, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_03": LodCandidate("prop_tree_lficus_03", 0.46875, 0.359375, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_05": LodCandidate("prop_tree_lficus_05", 0.46875, 0.3125, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            "prop_tree_lficus_06": LodCandidate("prop_tree_lficus_06", 0.453125, 0.43, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1), None, 0.15),
            "prop_tree_mquite_01": LodCandidate("prop_tree_mquite_01", 0.46875),
            "prop_rio_del_01": LodCandidate("prop_rio_del_01", 0.53125),
            "prop_rus_olive": LodCandidate("prop_rus_olive", 0.484375, 0.53125, UV(0, 0), UV(1, 0.5), UV(0, 0.5), UV(1, 1)),
            # bushes
            "prop_bush_lrg_04b": LodCandidate("prop_bush_lrg_04b", 0.37333333, 0.40625, UV(0, 0), UV(0.5859375, 0.5), UV(0, 0.5), UV(1, 1), None, None,
                UV(0.5859375, 0), UV(1, 0.5), 0.45283018867),
            "prop_bush_lrg_04c": LodCandidate("prop_bush_lrg_04c", 0.37333333, 0.4375, UV(0, 0), UV(0.5859375, 0.5), UV(0, 0.5), UV(1, 1), None, None,
                UV(0.5859375, 0), UV(1, 0.5), 0.5660377358),
            "prop_bush_lrg_04d": LodCandidate("prop_bush_lrg_04d", 0.4, 0.421875, UV(0, 0), UV(0.5078125, 0.5), UV(0, 0.5), UV(0.75, 1), None, None,
                UV(0.5078125, 0), UV(1, 0.484375), 0.47619047619),
            # palms
            "prop_palm_fan_02_b": LodCandidate("prop_palm_fan_02_b", 0.515625, 0.25625, UV(0, 0), UV(1, 0.625), UV(0, 1), UV(1, 0.625),
                UV(0.484375, 0.8125)),
            "prop_palm_fan_03_c": LodCandidate("prop_palm_fan_03_c", 0.5, 0.166666667, UV(0, 0), UV(1, 0.75), UV(0, 0.75), UV(1, 1)),
            "prop_palm_fan_03_d": LodCandidate("prop_palm_fan_03_d", 0.484375, 0.15865384615, UV(0, 0), UV(1, 0.8125), UV(0, 0.8125), UV(1, 1),
                UV(0.484375, 0.90625)),
            "prop_palm_fan_04_b": LodCandidate("prop_palm_fan_04_b", 0.484375, 0.321875, UV(0, 0), UV(1, 0.625), UV(0, 0.625), UV(1, 1),
                UV(0.46875, 0.80859375)),
            "prop_palm_huge_01a": LodCandidate("prop_palm_huge_01a", 0.46875, 0.11574074074, UV(0, 0), UV(1, 0.84375), UV(0, 1), UV(1, 0.84375),
                UV(0.484375, 0.921875)),
            "prop_palm_med_01b": LodCandidate("prop_palm_med_01b", 0.515625, 0.24431818181, UV(0, 0), UV(1, 0.6875), UV(0, 1), UV(1, 0.6875),
                UV(0.546875, 0.84375)),
            "prop_palm_sm_01e": LodCandidate("prop_palm_sm_01e", 0.515625, None, UV(0, 0), UV(1, 1)),
        }
        # add other Props that should use the same UV mapping
        lodCandidates["prop_palm_sm_01d"] = lodCandidates["prop_palm_sm_01f"] = \
            lodCandidates["prop_palm_sm_01e"]
        lodCandidates["prop_palm_med_01a"] = lodCandidates["prop_palm_med_01c"] = \
            lodCandidates["prop_palm_med_01b"]
        lodCandidates["prop_fan_palm_01a"] = lodCandidates["prop_palm_fan_02_a"] = \
            lodCandidates["prop_palm_fan_02_b"]
        lodCandidates["prop_palm_sm_01a"] = lodCandidates["prop_palm_fan_04_a"] = lodCandidates["prop_palm_fan_04_b"] = \
            lodCandidates["prop_palm_fan_04_b"]
        lodCandidates["prop_palm_fan_03_a"] = lodCandidates["prop_palm_fan_03_b"] = lodCandidates["prop_palm_fan_03_c_graff"] = lodCandidates["prop_palm_fan_04_c"] = \
            lodCandidates["prop_palm_fan_03_c"]
        lodCandidates["prop_palm_med_01d"] = lodCandidates["prop_palm_fan_03_d_graff"] = lodCandidates["prop_palm_fan_04_d"] = \
            lodCandidates["prop_palm_fan_03_d"]
        lodCandidates["prop_palm_huge_01b"] = \
            lodCandidates["prop_palm_huge_01a"]

        self.lodCandidates = lodCandidates

    def prepareSlodCandidates(self):
        slodCandidates = {
            # trees2
            "prop_tree_lficus_02": UVMap("trees2", UV(0 / 2, 0 / 4), UV(1 / 2, 1 / 4), UV(0 / 2, 0 / 4), UV(1 / 2, 1 / 4)),
            "prop_tree_lficus_03": UVMap("trees2", UV(1 / 2, 0 / 4), UV(2 / 2, 1 / 4), UV(1 / 2, 0 / 4), UV(2 / 2, 1 / 4)),
            "prop_tree_lficus_05": UVMap("trees2", UV(0 / 2, 1 / 4), UV(1 / 2, 2 / 4), UV(0 / 2, 1 / 4), UV(1 / 2, 2 / 4)),
            "prop_tree_lficus_06": UVMap("trees2", UV(1 / 2, 1 / 4), UV(2 / 2, 2 / 4), UV(1 / 2, 1 / 4), UV(2 / 2, 2 / 4)),
            "prop_tree_mquite_01": UVMap("trees2", UV(0 / 2, 2 / 4), UV(1 / 2, 3 / 4)),
            "prop_rio_del_01": UVMap("trees2", UV(1 / 2, 2 / 4), UV(2 / 2, 3 / 4)),
            "prop_rus_olive": UVMap("trees2", UV(0 / 2, 3 / 4), UV(1 / 2, 4 / 4), UV(0 / 2, 2 / 4), UV(1 / 2, 3 / 4)),
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
            "prop_tree_cedar_s_01": UVMap("trees", UV(6 / 16, 6 / 16), UV(8 / 16, 10 / 16), UV(0 / 16, 6 / 16), UV(3 / 16, 9 / 16)),
            "prop_tree_cedar_s_04": UVMap("trees", UV(3 / 16, 8 / 16), UV(4 / 16, 12 / 16), UV(3 / 16, 6 / 16), UV(3 / 16, 6 / 16)),
            "prop_tree_cedar_s_05": UVMap("trees", UV(3 / 16, 8 / 16), UV(4 / 16, 12 / 16)),
            "prop_tree_cedar_s_06": UVMap("trees", UV(6 / 16, 0 / 16), UV(7 / 16, 3 / 16)),
            "prop_tree_cypress_01": UVMap("trees", UV(4 / 16, 6 / 16), UV(6 / 16, 11 / 16)),
            "prop_tree_eng_oak_01": UVMap("trees", UV(10 / 16, 0 / 16), UV(13 / 16, 3 / 16), UV(9 / 16, 0 / 16), UV(12 / 16, 3 / 16)),
            "prop_tree_eucalip_01": UVMap("trees", UV(0 / 16, 8 / 16), UV(3 / 16, 12 / 16), UV(9 / 16, 3 / 16), UV(12 / 16, 6 / 16)),
            "prop_tree_fallen_pine_01": UVMap("trees", UV(12 / 16, 3 / 16), UV(14 / 16, 8 / 16), UV(0 / 16, 14 / 16), UV(3 / 16, 16 / 16)),
            "prop_tree_jacada_01": UVMap("trees", UV(0 / 16, 0 / 16), UV(3 / 16, 3 / 16), UV(0 / 16, 0 / 16), UV(3 / 16, 3 / 16)),
            "prop_tree_jacada_02": UVMap("trees", UV(3 / 16, 0 / 16), UV(6 / 16, 3 / 16), UV(3 / 16, 0 / 16), UV(6 / 16, 3 / 16)),
            "prop_tree_maple_02": UVMap("trees", UV(0 / 16, 3 / 16), UV(2 / 16, 5 / 16)),
            # TODO provide top view
            "prop_tree_maple_03": UVMap("trees", UV(0 / 16, 5 / 16), UV(2 / 16, 8 / 16)),
            "prop_tree_oak_01": UVMap("trees", UV(13 / 16, 0 / 16), UV(16 / 16, 3 / 16), UV(12 / 16, 0 / 16), UV(15 / 16, 3 / 16)),
            "prop_tree_olive_01": UVMap("trees", UV(7 / 16, 0 / 16), UV(10 / 16, 3 / 16), UV(6 / 16, 0 / 16), UV(9 / 16, 3 / 16)),
            "prop_tree_pine_01": UVMap("trees", UV(0 / 16, 12 / 16), UV(2 / 16, 16 / 16), UV(0 / 16, 9 / 16), UV(3 / 16, 12 / 16)),
            "prop_tree_pine_02": UVMap("trees", UV(2 / 16, 12 / 16), UV(4 / 16, 16 / 16), UV(3 / 16, 9 / 16), UV(6 / 16, 12 / 16)),
            "prop_w_r_cedar_01": UVMap("trees", UV(10 / 16, 11 / 16), UV(12 / 16, 16 / 16), UV(6 / 16, 9 / 16), UV(9 / 16, 12 / 16)),
            "prop_w_r_cedar_dead": UVMap("trees", UV(14 / 16, 3 / 16), UV(16 / 16, 8 / 16), UV(3 / 16, 12 / 16), UV(6 / 16, 15 / 16)),
            "test_tree_cedar_trunk_001": UVMap("trees", UV(12 / 16, 8 / 16), UV(14 / 16, 16 / 16), UV(9 / 16, 9 / 16), UV(12 / 16, 12 / 16)),
            "test_tree_forest_trunk_01": UVMap("trees", UV(14 / 16, 8 / 16), UV(16 / 16, 16 / 16), UV(12 / 16, 9 / 16), UV(15 / 16, 12 / 16)),
            "test_tree_forest_trunk_04": UVMap("trees", UV(2 / 16, 5 / 16), UV(4 / 16, 8 / 16), UV(12 / 16, 3 / 16), UV(15 / 16, 6 / 16)),
            # bushes
            "prop_bush_lrg_04b": UVMap("bushes", UV(0.5, 0), UV(1, 0.5), UV(0.5, 0), UV(1, 0.5)),
            "prop_bush_lrg_04c": UVMap("bushes", UV(0, 0.5), UV(0.5, 1), UV(0, 0.5), UV(0.5, 1)),
            "prop_bush_lrg_04d": UVMap("bushes", UV(0.5, 0.5), UV(1, 1), UV(0.5, 0.5), UV(1, 1)),
            # palms
            "prop_palm_sm_01e": UVMap("palms", UV(0 / 4, 0 / 4), UV(1 / 4, 2 / 4)),
            "prop_palm_fan_02_b": UVMap("palms", UV(0 / 4, 2 / 4), UV(1 / 4, 4 / 4), UV(0, 0), UV(0.5, 0.5), 0.23692810457),
            "prop_palm_fan_03_c": UVMap("palms", UV(1 / 4, 0 / 4), UV(2 / 4, 4 / 4), UV(0.5, 0), UV(1, 0.5), 0.14356435643),
            "prop_palm_fan_03_d": UVMap("palms", UV(2 / 4, 0 / 4), UV(3 / 4, 4 / 4), UV(0, 0.5), UV(0.5, 1), 0.13046937151),
            "prop_palm_huge_01a": UVMap("palms", UV(3 / 4, 0 / 4), UV(4 / 4, 4 / 4), UV(0.5, 0.5), UV(1, 1), 0.09644268774),
        }
        # add other Props that should use the same UV mapping
        slodCandidates["prop_palm_sm_01d"] = slodCandidates["prop_palm_sm_01f"] = slodCandidates["prop_palm_med_01a"] = slodCandidates["prop_palm_med_01b"] = \
            slodCandidates["prop_palm_med_01c"] = slodCandidates["prop_palm_sm_01e"]
        slodCandidates["prop_fan_palm_01a"] = slodCandidates["prop_palm_fan_02_a"] = slodCandidates["prop_palm_sm_01a"] = slodCandidates["prop_palm_fan_04_a"] = \
            slodCandidates["prop_palm_fan_04_b"] = slodCandidates["prop_palm_fan_02_b"]
        slodCandidates["prop_palm_fan_03_a"] = slodCandidates["prop_palm_fan_03_b"] = slodCandidates["prop_palm_fan_03_c_graff"] = slodCandidates[
            "prop_palm_fan_04_c"] = \
            slodCandidates["prop_palm_fan_03_c"]
        slodCandidates["prop_palm_med_01d"] = slodCandidates["prop_palm_fan_03_d_graff"] = slodCandidates["prop_palm_fan_04_d"] = \
            slodCandidates["prop_palm_fan_03_d"]
        slodCandidates["prop_palm_huge_01b"] = \
            slodCandidates["prop_palm_huge_01a"]

        self.slodCandidates = slodCandidates

    LOD_DISTANCE = 750  # the actual lodDistance for a lod entity is obtained by adding the furthest distance of all lod entities sharing the same slod model
    SLOD_DISTANCE = 1500  # somehow arbitrary and not that important because SLOD1 and SLOD2 are actually the same models.
    # however, reducing this results in smaller streaming extents of LOD/SLOD1 maps
    SLOD2_DISTANCE = 3000  # using 3000 because max height in game is 2600 and therefore until that height (plus a bit to allow slight xy offset)
    # a model with xy plane is needed so that objects don't vanish when above (SLOD2 models do contain such a xy plane)
    SLOD3_DISTANCE = 15000  # that seems to be the default value from Rockstars (in fact the whole map is not that large anyway)

    NUM_CHILDREN_MAX_VALUE = 255  # TODO confirm following claim: must be <= 255 since numChildren is of size 1 byte
    ENTITIES_EXTENTS_MAX_DIAGONAL_LOD = 100
    ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD1 = 400
    ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD2 = 800
    ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD3 = 1600

    unitBox = Box.createUnitBox()
    unitSphere = Sphere.createUnitSphere()

    # only entities with a lodDistance (according to hd entity) greater or equal this value are considered for SLOD1 to 3 model
    MIN_HD_LOD_DISTANCE_FOR_SLOD1 = Util.calculateLodDistance(unitBox, unitSphere, [4] * 3, True)  # 180
    MIN_HD_LOD_DISTANCE_FOR_SLOD2 = Util.calculateLodDistance(unitBox, unitSphere, [8] * 3, True)  # 240
    MIN_HD_LOD_DISTANCE_FOR_SLOD3 = Util.calculateLodDistance(unitBox, unitSphere, [13] * 3, True)  # 290

    TEXTURE_UV_EPS = 1 / 512

    def __init__(self, inputDir: str, outputDir: str, prefix: str):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.prefix = prefix
        self.slodYtypItems = None

    def run(self):
        print("running lod map creator...")
        self.determinePrefixBundles()
        self.readTemplates()
        self.prepareLodCandidates()
        self.prepareSlodCandidates()
        self.createOutputDir()
        self.readYtypItems()
        self.processFiles()
        self.fixMapExtents()
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
        if archetypeName in self.lodCandidates:
            index = mutableIndex[0]
            parentIndex = hdToLod[index] + offsetParentIndex
            mutableIndex[0] += 1
        else:
            parentIndex = -1

        return matchobj.group(1) + str(parentIndex) + matchobj.group(3)

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

    def replTranslateVertex(self, match: Match, translation: list[float]) -> str:
        vertex = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        newVertex = np.subtract(vertex, translation).tolist()
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

    def createSlodModel(self, nameWithoutSlodLevel: str, slodLevel: int, entities: list[EntityItem], parentIndex: int, numChildren: int, lodLevel: str, flags: int) -> EntityItem:
        name = nameWithoutSlodLevel + str(slodLevel)

        normalAndColorsFront = " / 0.00000000 -1.00000000 0.00000000 / 255 29 0 255 / 0 255 0 0 / "
        normalAndColorsTop = " / 0.00000000 0.00000000 1.00000000 / 255 29 0 255 / 0 0 0 0 / "
        verticesFront = {}
        verticesTop = {}
        numFrontPlanes = {}
        numTopPlanes = {}
        bbox = {}
        for entity in entities:
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

        totalBbox = Box.createReversedInfinityBox()

        for diffuseSampler in verticesFront:
            totalBbox.extendByPoint(bbox[diffuseSampler].min)
            totalBbox.extendByPoint(bbox[diffuseSampler].max)

        translation = totalBbox.getCenter()

        totalBbox = totalBbox.getTranslated(np.multiply(translation, [-1]).tolist())

        bounds = self.createAabb(totalBbox)

        shaders = ""
        geometries = ""
        shaderIndex = 0
        for diffuseSampler in verticesFront:
            indicesTopStr = self.createIndices(numTopPlanes[diffuseSampler])
            indicesFrontStr = self.createIndices(numFrontPlanes[diffuseSampler])

            bbox[diffuseSampler] = bbox[diffuseSampler].getTranslated(np.multiply(translation, [-1]).tolist())
            verticesFrontStr = re.sub('(?<=\t\t\t\t)(\\S+) (\\S+) (\\S+)', lambda match: self.replTranslateVertex(match, translation), verticesFront[diffuseSampler])
            verticesTopStr = re.sub('(?<=\t\t\t\t)(\\S+) (\\S+) (\\S+)', lambda match: self.replTranslateVertex(match, translation), verticesTop[diffuseSampler])

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
                shaders += self.contentTemplateOdrShaderTreeLod.replace("${DIFFUSE_SAMPLER}", diffuseSampler + "_top")

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
            .replace("${MESH_FILENAME}", os.path.join(os.path.relpath(self.getOutputDirMeshes(), self.getOutputDirModels()), name.lower() + ".mesh")) \
            .replace("${SHADERS}\n", shaders)

        fileModelOdr = open(os.path.join(self.getOutputDirModels(), name.lower() + ".odr"), 'w')
        fileModelOdr.write(contentModelOdr)
        fileModelOdr.close()

        if slodLevel == 1:
            itemHdDistance = LodMapCreator.LOD_DISTANCE
            itemLodDistance = LodMapCreator.SLOD_DISTANCE
        elif slodLevel == 2:
            itemHdDistance = LodMapCreator.SLOD_DISTANCE
            itemLodDistance = LodMapCreator.SLOD2_DISTANCE
        elif slodLevel == 3:
            itemHdDistance = LodMapCreator.SLOD2_DISTANCE
            itemLodDistance = LodMapCreator.SLOD3_DISTANCE
        elif slodLevel == 0:    # Wieder Ausbauen, nur vorübergehend bis eigene Methode zum erzeugen der LOD-Modelle vorhanden ist
            itemHdDistance = 350
            itemLodDistance = LodMapCreator.LOD_DISTANCE
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

        return EntityItem(name, translation, [1, 1, 1], [1, 0, 0, 0], itemLodDistance, itemHdDistance, parentIndex, numChildren, lodLevel, flags)

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
        hdEntitiesWithLod = []
        lodCoords = []
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
            contentNoLod = Ymap.calculateAndReplaceLodDistanceForEntitiesWithLod(contentNoLod, self.ytypItems)

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
                if archetypeName not in self.lodCandidates:
                    continue

                position = [float(matchobj.group(2)), float(matchobj.group(3)), float(matchobj.group(4))]
                rotation = [float(matchobj.group(8)), float(matchobj.group(5)), float(matchobj.group(6)), float(matchobj.group(7))]
                scale = [float(matchobj.group(9)), float(matchobj.group(9)), float(matchobj.group(10))]
                lodDistance = float(matchobj.group(11))
                entity = EntityItem(archetypeName, position, scale, rotation, lodDistance)

                hdEntitiesWithLod.append(entity)

                lodCoords.append(position)

        hierarchy = self.calculateLodHierarchy(lodCoords)

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

            if hdEntity.lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD1:
                if h[0] not in lodToSlod1:
                    lodToSlod1[h[0]] = h[1]
                    slod1NumChildren[h[1]] = slod1NumChildren.get(h[1], 0) + 1

                if h[1] not in entitiesForSlod1Models:
                    entitiesForSlod1Models[h[1]] = []
                entitiesForSlod1Models[h[1]].append(hdEntity)
            if hdEntity.lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD2:
                if h[1] not in slod1ToSlod2:
                    slod1ToSlod2[h[1]] = h[2]
                    slod2NumChildren[h[2]] = slod2NumChildren.get(h[2], 0) + 1

                if h[2] not in entitiesForSlod2Models:
                    entitiesForSlod2Models[h[2]] = []
                entitiesForSlod2Models[h[2]].append(hdEntity)
            if hdEntity.lodDistance >= LodMapCreator.MIN_HD_LOD_DISTANCE_FOR_SLOD3:
                if h[2] not in slod2ToSlod3:
                    slod2ToSlod3[h[2]] = h[3]
                    slod3NumChildren[h[3]] = slod3NumChildren.get(h[3], 0) + 1

                if h[3] not in entitiesForSlod3Models:
                    entitiesForSlod3Models[h[3]] = []
                entitiesForSlod3Models[h[3]].append(hdEntity)

        lodEntities = []
        slod1Entities = []
        slod2Entities = []
        slod3Entities = []

        index = 0
        slod3KeyToIndex = {}
        for key in sorted(entitiesForSlod3Models):
            slodName = mapPrefix.lower() + "_" + str(index) + "_slod"
            slod3Entities.append(self.createSlodModel(
                slodName, 3,
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
                slodName, 2,
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
            slod1Entities.append(self.createSlodModel(
                slodName, 1,
                entitiesForSlod1Models[key],
                parentIndex, slod1NumChildren[key],
                LodLevel.SLOD1, Flag.FLAGS_SLOD1
            ))
            slod1KeyToIndex[key] = index
            index += 1

        for key in sorted(entitiesForLodModels):
            lodName = mapPrefix.lower() + "_" + str(key) + "_lod"
            parentIndex = self.getParentIndexForKey(key, lodToSlod1, slod1KeyToIndex, 0)
            lodEntities.append(self.createSlodModel(
                lodName, 0,
                entitiesForLodModels[key],
                parentIndex, lodNumChildren[key],
                LodLevel.LOD, Flag.FLAGS_LOD
            ))

        if len(slod2Entities) == 0:
            slod2MapName = None
        else:
            slod2MapName = mapPrefix.lower() + "_slod2"
            self.writeLodOrSlodMap(slod2MapName, None, ContentFlag.SLOD + ContentFlag.SLOD2, slod3Entities + slod2Entities)
        self.writeLodOrSlodMap(mapPrefix.lower() + "_lod", slod2MapName, ContentFlag.LOD + ContentFlag.SLOD, slod1Entities + lodEntities)

        self.adaptHdMapsForPrefix(mapPrefix, hdToLod, len(slod1Entities))

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

    def calculateLodHierarchy(self, points: list[list[float]]) -> list[list[int]]:
        if len(points) == 0:
            return []

        hierarchy = []
        for i in range(len(points)):
            hierarchy.append([])

        return self._calculateLodHierarchy(points, hierarchy)

    def _calculateLodHierarchy(self, points: list[list[float]], hierarchy: list[list[int]]) -> list[list[int]]:
        level = len(hierarchy[0])
        if level == 0:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD3
        elif level == 1:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD2
        elif level == 2:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_SLOD1
        elif level == 3:
            maxExtends = LodMapCreator.ENTITIES_EXTENTS_MAX_DIAGONAL_LOD
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
            pointsOfParent[parentIndex].append(points[i])

        clusterOffset = 0
        for parentIndex in range(len(pointsOfParent)):
            clustering, unused = Util.performClustering(pointsOfParent[parentIndex], LodMapCreator.NUM_CHILDREN_MAX_VALUE if level == 3 else -1, maxExtends)

            for c in range(len(clustering)):
                i = absIndices[parentIndex][c]
                hierarchy[i].insert(0, clustering[c] + clusterOffset)

            clusterOffset += max(clustering) + 1

        return self._calculateLodHierarchy(points, hierarchy)

    def copyOthers(self):
        # copy other files
        Util.copyFiles(self.inputDir, self.getOutputDirMaps(),
            lambda filename: not filename.endswith(".ymap.xml") or not filename.startswith(tuple(each.lower() for each in self.bundlePrefixes)))

    # adapt extents and set current datetime
    def fixMapExtents(self):
        print("\tfixing map extents")

        if self.slodYtypItems is not None:
            self.ytypItems |= YtypParser.readYtypDirectory(self.getOutputDirModels())

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

    def copyTextureDictionaries(self):
        texturesDir = os.path.join(os.path.dirname(__file__), "textures")
        for filename in os.listdir(texturesDir):
            shutil.copyfile(os.path.join(texturesDir, filename), os.path.join(self.getOutputDirModels(), self.prefix + "_" + filename))
