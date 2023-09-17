import re

from common.BoundingGeometry import BoundingGeometry
from common.Util import Util


class Capsule:
    centerTop: int
    centerBottom: int
    radius: float
    materialIndex: int

    @staticmethod
    def parse(content: str) -> "Capsule":
        centerTop = -1
        centerBottom = -1
        radius = -1
        materialIndex = -1
        i = -1
        for line in content.splitlines():
            i += 1
            if i == 0:
                m = re.match(r'				Capsule \d+$', line)
                if m is not None:
                    continue
            elif i == 1 and line == "				{":
                continue
            elif i == 2:
                m = re.match(r'					CenterTop (\d+)$', line)
                if m is not None:
                    centerTop = int(m.group(1))
                    continue
            elif i == 3:
                m = re.match(r'					CenterBottom (\d+)$', line)
                if m is not None:
                    centerBottom = int(m.group(1))
                    continue
            elif i == 4:
                m = re.match(r'					Radius (\d+\.\d+)$', line)
                if m is not None:
                    # TODO apply matrix to radius
                    radius = float(m.group(1))
                    continue
            elif i == 5:
                m = re.match(r'					MaterialIndex (\d+)$', line)
                if m is not None:
                    materialIndex = int(m.group(1))
                    continue
            elif i == 6 and line == "				}":
                continue
            elif i == 7 and line == "":
                continue

            raise Exception("Could not parse Capsule. Error in line " + str(i + 1) + ":\n" + content)

        return Capsule(centerTop, centerBottom, radius, materialIndex)

    def __init__(self, centerTop: int, centerBottom: int, radius: float, materialIndex: int):
        self.centerTop = centerTop
        self.centerBottom = centerBottom
        self.radius = radius
        self.materialIndex = materialIndex

    def offsetVertexIndex(self, offsetVertex: int, offsetPolygon: int) -> None:
        self.centerTop = self.centerTop + offsetVertex
        self.centerBottom = self.centerBottom + offsetVertex

    def scale(self, scale: float) -> None:
        self.radius = self.radius * scale

    def asPolygonString(self, polygonIndex: int) -> str:
        return """				Capsule """ + str(polygonIndex) + """
				{
					CenterTop """ + str(self.centerTop) + """
					CenterBottom """ + str(self.centerBottom) + """
					Radius """ + Util.floatToStr(self.radius) + """
					MaterialIndex """ + str(self.materialIndex) + """
				}
"""

    def extendBoundingGeometry(self, boundingGeometry: BoundingGeometry, vertices: list[list[float]]):
        centerTop = vertices[self.centerTop]
        centerBottom = vertices[self.centerBottom]

        boundingGeometry.extendBySphere(centerTop, self.radius)
        boundingGeometry.extendBySphere(centerBottom, self.radius)
