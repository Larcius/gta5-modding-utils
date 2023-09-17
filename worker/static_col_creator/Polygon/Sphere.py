import re

from common.BoundingGeometry import BoundingGeometry
from common.Util import Util


class Sphere:
    center: int
    radius: float
    materialIndex: int

    @staticmethod
    def parse(content: str) -> "Sphere":
        center = -1
        radius = -1
        materialIndex = -1
        i = -1
        for line in content.splitlines():
            i += 1
            if i == 0:
                m = re.match(r'				Sphere \d+$', line)
                if m is not None:
                    continue
            elif i == 1 and line == "				{":
                continue
            elif i == 2:
                m = re.match(r'					Center (\d+)$', line)
                if m is not None:
                    center = int(m.group(1))
                    continue
            elif i == 3:
                m = re.match(r'					Radius (\d+\.\d+)$', line)
                if m is not None:
                    # TODO apply matrix to radius
                    radius = float(m.group(1))
                    continue
            elif i == 4:
                m = re.match(r'					MaterialIndex (\d+)$', line)
                if m is not None:
                    materialIndex = int(m.group(1))
                    continue
            elif i == 5 and line == "				}":
                continue
            elif i == 6 and line == "":
                continue

            raise Exception("Could not parse Sphere. Error in line " + str(i + 1) + ":\n" + content)

        return Sphere(center, radius, materialIndex)

    def __init__(self, center: int, radius: float, materialIndex: int):
        self.center = center
        self.radius = radius
        self.materialIndex = materialIndex

    def offsetVertexIndex(self, offsetVertex: int, offsetPolygon: int) -> None:
        self.center = self.center + offsetVertex

    def scale(self, scale: float) -> None:
        self.radius = self.radius * scale

    def asPolygonString(self, polygonIndex: int) -> str:
        return """				Sphere """ + str(polygonIndex) + """
				{
					Center """ + str(self.center) + """
					Radius """ + Util.floatToStr(self.radius) + """
					MaterialIndex """ + str(self.materialIndex) + """
				}
"""

    def extendBoundingGeometry(self, boundingGeometry: BoundingGeometry, vertices: list[list[float]]):
        center = vertices[self.center]
        boundingGeometry.extendBySphere(center, self.radius)
