from typing import Optional
from common.texture.UV import UV


class LodCandidate:
    name: str
    texture_origin: float
    planeZ: Optional[float]
    uvFrontMin: UV
    uvFrontMax: UV
    uvTopMin: Optional[UV]
    uvTopMax: Optional[UV]
    uvTopCenter: Optional[UV]
    uvTopCenterZ: Optional[float]
    uvSideMin: Optional[UV]
    uvSideMax: Optional[UV]
    _textureOriginSide: Optional[float]

    def __init__(self, name: str, texture_origin: float = 0.5, planeZ: Optional[float] = 0.5,
            uvFrontMin: UV = UV(0, 0), uvFrontMax: UV = UV(1, 1),
            uvTopMin: Optional[UV] = None, uvTopMax: Optional[UV] = None, uvTopCenter: Optional[UV] = None, uvTopCenterZ: Optional[float] = None,
            uvSideMin: Optional[UV] = None, uvSideMax: Optional[UV] = None, _textureOriginSide: Optional[float] = None):
        self.name = name
        self.texture_origin = texture_origin
        self.planeZ = planeZ
        self.uvFrontMin = uvFrontMin
        self.uvFrontMax = uvFrontMax
        self.uvTopMin = uvTopMin
        self.uvTopMax = uvTopMax
        self.uvTopCenter = uvTopCenter
        self.uvTopCenterZ = uvTopCenterZ
        self.uvSideMin = uvSideMin
        self.uvSideMax = uvSideMax
        self._textureOriginSide = _textureOriginSide

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
