from typing import Optional

from common.Box import Box
from common.texture.UV import UV


class LodCandidate:
    # needed to avoid getting wrong pixels from texture (e.g. using 0 would result in getting right-most resp. down-most pixel)
    TEXTURE_UV_EPS = 1 / 256

    name: str
    texture_origin: float
    planeZ: float
    uvFrontMin: UV
    uvFrontMax: UV
    uvTopMin: Optional[UV]
    uvTopMax: Optional[UV]
    uvTopCenter: Optional[UV]
    uvSideMin: Optional[UV]
    uvSideMax: Optional[UV]
    _textureOriginSide: Optional[float]

    def __init__(self, name: str, texture_origin: float = 0.5, planeZ: Optional[float] = 0.5,
            uvFrontMin: UV = UV(0, 0), uvFrontMax: UV = UV(1, 1),
            uvTopMin: Optional[UV] = None, uvTopMax: Optional[UV] = None, uvTopCenter: Optional[UV] = None,
            uvSideMin: Optional[UV] = None, uvSideMax: Optional[UV] = None, textureOriginSide: Optional[float] = None):
        self.name = name
        self.texture_origin = texture_origin
        self.planeZ = planeZ
        self.uvFrontMin = uvFrontMin
        self.uvFrontMax = uvFrontMax
        self.uvTopMin = uvTopMin
        self.uvTopMax = uvTopMax
        self.uvTopCenter = uvTopCenter
        self.uvSideMin = uvSideMin
        self.uvSideMax = uvSideMax
        self._textureOriginSide = textureOriginSide

    @staticmethod
    def createTextureUvWithEps(minUv: UV, maxUv: UV) -> (UV, UV):
        if minUv is None or maxUv is None:
            raise Exception("minUv and maxUv must not be None")

        minUvEps = UV(minUv.u, minUv.v)
        maxUvEps = UV(maxUv.u, maxUv.v)
        if minUvEps.u < maxUvEps.u:
            minUvEps.u += LodCandidate.TEXTURE_UV_EPS
        else:
            maxUvEps.u += LodCandidate.TEXTURE_UV_EPS

        if minUvEps.v < maxUvEps.v:
            minUvEps.v += LodCandidate.TEXTURE_UV_EPS
        else:
            maxUvEps.v += LodCandidate.TEXTURE_UV_EPS

        return minUvEps, maxUvEps

    def getUvFrontMin(self) -> UV:
        return LodCandidate.createTextureUvWithEps(self.uvFrontMin, self.uvFrontMax)[0]

    def getUvFrontMax(self) -> UV:
        return LodCandidate.createTextureUvWithEps(self.uvFrontMin, self.uvFrontMax)[1]

    def getUvSideMin(self) -> UV:
        if self.hasDedicatedSideTexture():
            return LodCandidate.createTextureUvWithEps(self.uvSideMin, self.uvSideMax)[0]
        else:
            return self.getUvFrontMin()

    def getUvSideMax(self) -> UV:
        if self.hasDedicatedSideTexture():
            return LodCandidate.createTextureUvWithEps(self.uvSideMin, self.uvSideMax)[1]
        else:
            return self.getUvFrontMax()

    def getUvTopMin(self) -> UV:
        return LodCandidate.createTextureUvWithEps(self.uvTopMin, self.uvTopMax)[0]

    def getUvTopMax(self) -> UV:
        return LodCandidate.createTextureUvWithEps(self.uvTopMin, self.uvTopMax)[1]

    def textureOriginSide(self) -> Optional[float]:
        return self.texture_origin if self._textureOriginSide is None else self._textureOriginSide

    def texture(self) -> str:
        return "lod_" + self.name.lower()

    def hasTop(self, boundingBox: Box, scale: list[float]) -> bool:
        if self.uvTopMin is None or self.uvTopMax is None:
            return False

        scaledBoundingBox = boundingBox.getScaled(scale)
        return scaledBoundingBox.getDiagonalOfPlaneXY() >= 7

    def hasDiagonal(self, boundingBox: Box, scale: list[float]) -> bool:
        scaledBoundingBox = boundingBox.getScaled(scale)
        return scaledBoundingBox.getDiagonalOfPlaneXY() >= 12 or scaledBoundingBox.getDiagonal() >= 40

    def hasDedicatedSideTexture(self) -> bool:
        return self.uvSideMin is not None and self.uvSideMax is not None

    def getUvTopCenter(self):
        if self.uvTopCenter is None:
            minUv = UV(min(self.uvTopMin.u, self.uvTopMax.u), min(self.uvTopMin.v, self.uvTopMax.v))
            maxUv = UV(max(self.uvTopMin.u, self.uvTopMax.u), max(self.uvTopMin.v, self.uvTopMax.v))
            if self.uvTopMin.u > self.uvTopMax.u:
                return UV(minUv.v + (maxUv.v - minUv.v) * (1 - self.texture_origin), minUv.u + (maxUv.u - minUv.u) * self.textureOriginSide())
            else:
                return UV(minUv.u + (maxUv.u - minUv.u) * self.texture_origin, minUv.v + (maxUv.v - minUv.v) * (1 - self.textureOriginSide()))
        else:
            return self.uvTopCenter

    def getTextureDictionary(self, prefix: str) -> str:
        if self.name.startswith("prop_bush_"):
            dictMain = "bushes"
        elif self.name.startswith("prop_palm_") or self.name.startswith("prop_fan_palm_"):
            dictMain = "palms"
        else:
            dictMain = "trees"
        return prefix + "_" + dictMain + "_lod"
