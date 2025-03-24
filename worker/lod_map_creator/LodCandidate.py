from typing import Optional

from common.Box import Box
from common.texture.UV import UV


class LodCandidate:
    diffuseSampler: str
    textureOrigin: float
    planeZ: float
    uvFrontMin: UV
    uvFrontMax: UV
    uvTopMin: Optional[UV]
    uvTopMax: Optional[UV]
    uvTopCenter: Optional[UV]
    uvSideMin: Optional[UV]
    uvSideMax: Optional[UV]
    _textureOriginSide: Optional[float]
    sideOffsetZ: float

    def __init__(self, textureOrigin: float = 0.5, planeZ: Optional[float] = 0.5,
            uvFrontMin: UV = UV(0, 0), uvFrontMax: UV = UV(1, 1),
            uvTopMin: Optional[UV] = None, uvTopMax: Optional[UV] = None, uvTopCenter: Optional[UV] = None,
            uvSideMin: Optional[UV] = None, uvSideMax: Optional[UV] = None, textureOriginSide: Optional[float] = None, sideOffsetZ: float = 0):
        self.textureOrigin = textureOrigin
        self.planeZ = planeZ
        self.uvFrontMin = uvFrontMin
        self.uvFrontMax = uvFrontMax
        self.uvTopMin = uvTopMin
        self.uvTopMax = uvTopMax
        self.uvTopCenter = uvTopCenter
        self.uvSideMin = uvSideMin
        self.uvSideMax = uvSideMax
        self._textureOriginSide = textureOriginSide
        self.sideOffsetZ = sideOffsetZ

    def getUvSideMin(self) -> UV:
        if self.hasDedicatedSideTexture():
            return self.uvSideMin
        else:
            return self.uvFrontMin

    def getUvSideMax(self) -> UV:
        if self.hasDedicatedSideTexture():
            return self.uvSideMax
        else:
            return self.uvFrontMax

    def textureOriginSide(self) -> Optional[float]:
        return self.textureOrigin if self._textureOriginSide is None else self._textureOriginSide

    def hasTop(self, boundingBox: Box, scale: list[float]) -> bool:
        if self.uvTopMin is None or self.uvTopMax is None:
            return False

        scaledBoundingBox = boundingBox.getScaled(scale)
        return scaledBoundingBox.getDiagonalOfPlaneXY() >= 7

    def hasDiagonal(self, boundingBox: Box, scale: list[float]) -> bool:
        if not self.hasDedicatedSideTexture():
            return False

        scaledBoundingBox = boundingBox.getScaled(scale)
        return scaledBoundingBox.getDiagonalOfPlaneXY() >= 25

    def hasDedicatedSideTexture(self) -> bool:
        return self.uvSideMin is not None and self.uvSideMax is not None

    def getUvTopCenter(self):
        if self.uvTopCenter is None:
            minUv = UV(min(self.uvTopMin.u, self.uvTopMax.u), min(self.uvTopMin.v, self.uvTopMax.v))
            maxUv = UV(max(self.uvTopMin.u, self.uvTopMax.u), max(self.uvTopMin.v, self.uvTopMax.v))
            if self.uvTopMin.u > self.uvTopMax.u:
                return UV(minUv.v + (maxUv.v - minUv.v) * (1 - self.textureOrigin), minUv.u + (maxUv.u - minUv.u) * self.textureOriginSide())
            else:
                return UV(minUv.u + (maxUv.u - minUv.u) * self.textureOrigin, minUv.v + (maxUv.v - minUv.v) * (1 - self.textureOriginSide()))
        else:
            return self.uvTopCenter
