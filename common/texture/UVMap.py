from common.texture.UV import UV


class UVMap:
    diffuseSamplerSuffix: str
    frontMin: UV
    frontMax: UV
    topMin: UV
    topMax: UV
    topZ: float

    def __init__(self, diffuseSamplerSuffix: str, frontMin: UV, frontMax: UV, topMin: UV = None, topMax: UV = None, topZ: float = None):
        self.diffuseSamplerSuffix = diffuseSamplerSuffix
        self.frontMin = frontMin
        self.frontMax = frontMax
        self.topMin = topMin
        self.topMax = topMax
        self.topZ = topZ

    def getDiffuseSampler(self):
        return "slod_" + self.diffuseSamplerSuffix
