import re

from common.Box import Box


class Tri:
    vertices: list[int]
    siblings: list[int]
    materialIndex: int

    @staticmethod
    def parse(content: str) -> "Tri":
        vertices = []
        siblings = []
        materialIndex = -1
        i = -1
        for line in content.splitlines():
            i += 1
            if i == 0:
                m = re.match(r'				Tri \d+', line)
                if m is not None:
                    continue
            elif i == 1 and line == "				{":
                continue
            elif i == 2:
                m = re.match(r'					Vertices (\d+) (\d+) (\d+)', line)
                if m is not None:
                    vertices = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
                    continue
            elif i == 3:
                m = re.match(r'					Siblings (-?\d+) (-?\d+) (-?\d+)', line)
                if m is not None:
                    siblings = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
                    continue
            elif i == 4:
                m = re.match(r'					MaterialIndex (\d+)', line)
                if m is not None:
                    materialIndex = int(m.group(1))
                    continue
            elif i == 5 and line == "				}":
                continue
            elif i == 6 and line == "":
                continue

            raise Exception("Could not parse Tri. Error in line " + str(i + 1) + ":\n" + content)

        return Tri(vertices, siblings, materialIndex)

    def __init__(self, vertices: list[int], siblings: list[int], materialIndex: int):
        if len(vertices) != 3:
            raise Exception("Expecting 3 vertices")
        if len(siblings) != 3:
            raise Exception("Expecting 3 siblings")

        self.vertices = vertices
        self.siblings = siblings
        self.materialIndex = materialIndex

    def offsetVertexIndex(self, offset: int) -> None:
        for i in range(len(self.vertices)):
            self.vertices[i] = self.vertices[i] + offset
        for i in range(len(self.siblings)):
            self.siblings[i] = self.siblings[i] + offset

    def scale(self, scale: float) -> None:
        pass

    def asPolygonString(self, polygonIndex: int) -> str:
        return """				Tri """ + str(polygonIndex) + """
				{
					Vertices """ + str(self.vertices[0]) + " " + str(self.vertices[1]) + " " + str(self.vertices[2]) + """
					Siblings """ + str(self.siblings[0]) + " " + str(self.siblings[1]) + " " + str(self.siblings[2]) + """
					MaterialIndex """ + str(self.materialIndex) + """
				}
"""

    def extendBoundingBox(self, bbox: Box, vertices: list[list[float]]):
        for index in self.vertices:
            bbox.extendByPoint(vertices[index])
