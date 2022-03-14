from typing import Optional
from common.texture.UV import UV


class UVMap:
    diffuseSamplerSuffix: str
    frontMin: UV
    frontMax: UV
    topMin: Optional[UV]
    topMax: Optional[UV]
    topZ: Optional[float]

    def __init__(self, diffuseSamplerSuffix: str, frontMin: UV, frontMax: UV, topMin: Optional[UV] = None, topMax: Optional[UV] = None, topZ: Optional[float] = None):
        self.diffuseSamplerSuffix = diffuseSamplerSuffix
        self.frontMin = frontMin
        self.frontMax = frontMax
        self.topMin = topMin
        self.topMax = topMax
        self.topZ = topZ

    def getDiffuseSampler(self):
        return "slod_" + self.diffuseSamplerSuffix
