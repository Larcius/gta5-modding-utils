import re

from common.BoundingGeometry import BoundingGeometry


class Box:
    vertices: list[int]
    materialIndex: int

    @staticmethod
    def parse(content: str) -> "Box":
        vertices = []
        materialIndex = -1
        i = -1
        for line in content.splitlines():
            i += 1
            if i == 0:
                m = re.match(r'				Box \d+$', line)
                if m is not None:
                    continue
            elif i == 1 and line == "				{":
                continue
            elif i == 2:
                m = re.match(r'					Vertices (\d+) (\d+) (\d+) (\d+)$', line)
                if m is not None:
                    vertices = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
                    continue
            elif i == 3:
                m = re.match(r'					MaterialIndex (\d+)$', line)
                if m is not None:
                    materialIndex = int(m.group(1))
                    continue
            elif i == 4 and line == "				}":
                continue
            elif i == 5 and line == "":
                continue

            raise Exception("Could not parse Box. Error in line " + str(i + 1) + ":\n" + content)

        return Box(vertices, materialIndex)

    def __init__(self, vertices: list[int], materialIndex: int):
        if len(vertices) != 4:
            raise Exception("Expecting 4 vertices")
        self.vertices = vertices
        self.materialIndex = materialIndex

    def offsetVertexIndex(self, offsetVertex: int, offsetPolygon: int) -> None:
        for i in range(len(self.vertices)):
            self.vertices[i] = self.vertices[i] + offsetVertex

    def scale(self, scale: float) -> None:
        pass

    def asPolygonString(self, polygonIndex: int) -> str:
        return """				Box """ + str(polygonIndex) + """
				{
					Vertices """ + str(self.vertices[0]) + " " + str(self.vertices[1]) + " " + str(self.vertices[2]) + " " + str(self.vertices[3]) + """
					MaterialIndex """ + str(self.materialIndex) + """
				}
"""

    def extendBoundingGeometry(self, boundingGeometry: BoundingGeometry, vertices: list[list[float]]):
        for index in self.vertices:
            boundingGeometry.extendByPoint(vertices[index])
