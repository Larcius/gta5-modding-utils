import re
from typing import Any, IO, Union

import numpy as np

from common.Box import Box as BBox
from common.Util import Util
from worker.static_col_creator.Material import Material
from worker.static_col_creator.Polygon.Box import Box
from worker.static_col_creator.Polygon.Capsule import Capsule
from worker.static_col_creator.Polygon.Cylinder import Cylinder
from worker.static_col_creator.Polygon.Sphere import Sphere
from worker.static_col_creator.Polygon.Tri import Tri


class BoundBVH:

    @staticmethod
    def parse(contentPhBound: str, contentMatrix: str, contentChildFlagsItem: str) -> "BoundBVH":
        matrix = BoundBVH.parseMatrix(contentMatrix)
        polygons, materials, margin, vertices, shrunk = BoundBVH.parsePhBound(contentPhBound, matrix)
        flags1, flags2 = BoundBVH.parseChildFlagsItem(contentChildFlagsItem)
        return BoundBVH(polygons, materials, margin, vertices, shrunk, flags1, flags2)

    @staticmethod
    def parsePhBound(contentPhBound: str, matrix: list[list[float]]) -> (list[Any], list[str], float, list[list[float]], Union[list[list[float]], None]):
        # modes:
        #  0: start of phBound (before Polygons)
        #  1: start of Polygons
        #  2: within Polygons
        #  3: start of single polygon
        #  4: within single polygon
        #  5: after polygons expecting GeometryCenter
        #  6: after GeometryCenter expecting Vertices
        #  7: start of Vertices
        #  8: within Vertices
        #  9: after Vertices expecting VertexColors
        # 10: after VertexColors expecting Materials
        # 11: start of Materials
        # 12: within Materials
        # 13: start of single Material
        # 14: within single Material
        # 15: after Material expecting MaterialColors
        # 16: after MaterialColors expecting Margin
        # 17: after Margin expecting Shrunk or end of phBound
        # 18: start of Shrunk
        # 19: within Shrunk
        # 20: after Shrunk expecting end of phBound
        # 21: end of phBound expecting no more content (except empty line)
        mode = 0

        i = -1
        polygon = ""
        polygons = []
        geometryCenter = None
        vertices = []
        material = ""
        materials = []
        margin = None
        shrunk = None
        for line in contentPhBound.splitlines():
            i += 1
            if i == 0 and line == "		phBound":
                continue
            elif i == 1 and line == "		{":
                continue
            elif i == 2:
                m = re.match(r'			Type (?:BoundBVH|BoundGeometry)', line)
                if m is not None:
                    continue
            elif i == 3:
                m = re.match(r'			Radius ([+-]?\d+\.\d+)', line)
                if m is not None:
                    continue
            elif 4 <= i <= 7:
                m = re.match(r'			(?:AABBMax|AABBMin|Centroid|CG) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)', line)
                if m is not None:
                    continue

            # Polygons
            elif i == 8:
                m = re.match(r'			Polygons \d+', line)
                if m is not None:
                    mode = 1
                    continue
            elif mode == 1:
                if line == "			{":
                    mode = 2
                    continue
            elif mode == 2:
                m = re.match(r'				[a-zA-Z]+ \d+', line)
                if m is not None:
                    polygon = line + "\n"
                    mode = 3
                    continue
                elif line == "			}":
                    mode = 5
                    continue
            elif mode == 3:
                if line == "				{":
                    polygon = polygon + line + "\n"
                    mode = 4
                    continue
            elif mode == 4:
                if line.startswith("					"):
                    polygon = polygon + line + "\n"
                    continue
                elif line == "				}":
                    polygon = polygon + line + "\n"
                    polygons.append(BoundBVH.parsePolygon(polygon))
                    mode = 2
                    continue

            # GeometryCenter
            elif mode == 5:
                m = re.match(r'			GeometryCenter ([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)', line)
                if m is not None:
                    geometryCenter = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                    mode = 6
                    continue

            # Vertices
            elif mode == 6:
                m = re.match(r'			Vertices \d+', line)
                if m is not None:
                    mode = 7
                    continue
            elif mode == 7:
                if line == "			{":
                    mode = 8
                    continue
            elif mode == 8:
                m = re.match(r'				([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)', line)
                if m is not None:
                    vertex = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                    vertices.append(BoundBVH.transformVertex(vertex, geometryCenter, matrix))
                    continue
                if line == "			}":
                    mode = 9
                    continue

            # VertexColors
            elif mode == 9 and line == "			VertexColors null":
                mode = 10
                continue

            # Materials
            elif mode == 10:
                m = re.match(r'			Materials \d+', line)
                if m is not None:
                    mode = 11
                    continue
            elif mode == 11:
                if line == "			{":
                    mode = 12
                    continue
            elif mode == 12:
                m = re.match(r'				Material \d+', line)
                if m is not None:
                    material = line + "\n"
                    mode = 13
                    continue
                elif line == "			}":
                    mode = 15
                    continue
            elif mode == 13:
                if line == "				{":
                    material = material + line + "\n"
                    mode = 14
                    continue
            if mode == 14:
                if line.startswith("					"):
                    material = material + line + "\n"
                    continue
                elif line == "				}":
                    material = material + line + "\n"
                    materials.append(Material.parse(material))
                    mode = 12
                    continue

            # MaterialColors
            elif mode == 15 and line == "			MaterialColors null":
                mode = 16
                continue

            # Margin
            elif mode == 16:
                m = re.match(r'			Margin (\d+\.\d+)', line)
                if m is not None:
                    margin = float(m.group(1))
                    mode = 17
                    continue

            # Shrunk
            elif mode == 17:
                m = re.match(r'			Shrunk \d+', line)
                if m is not None:
                    shrunk = []
                    mode = 18
                    continue
            elif mode == 18:
                if line == "			{":
                    mode = 19
                    continue
            elif mode == 19:
                m = re.match(r'				([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)', line)
                if m is not None:
                    vertex = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                    shrunk.append(BoundBVH.transformVertex(vertex, geometryCenter, matrix))
                    continue
                if line == "			}":
                    mode = 20
                    continue

            # end of phBound
            if mode == 17 or mode == 20:
                if line == "		}":
                    mode = 21
                    continue
            elif mode == 21 and line == "":
                continue

            raise Exception("Could not parse phBound. Error in line " + str(i + 1) + ":\n" + contentPhBound)

        return polygons, materials, margin, vertices, shrunk

    @staticmethod
    def transformVertex(vertex: list[float], geometryCenter: list[float], matrix: list[list[float]]) -> list[float]:
        vec4PreTranslation = [vertex[0], vertex[1], vertex[2], 1]
        vector = np.dot(vec4PreTranslation, matrix)
        return [vector[0] + geometryCenter[0], vector[1] + geometryCenter[1], vector[2] + geometryCenter[2]]

    @staticmethod
    def parsePolygon(contentPolygon: str):
        if contentPolygon.startswith("				Box "):
            return Box.parse(contentPolygon)
        elif contentPolygon.startswith("				Capsule "):
            return Capsule.parse(contentPolygon)
        elif contentPolygon.startswith("				Cylinder "):
            return Cylinder.parse(contentPolygon)
        elif contentPolygon.startswith("				Sphere "):
            return Sphere.parse(contentPolygon)
        elif contentPolygon.startswith("				Tri "):
            return Tri.parse(contentPolygon)

    @staticmethod
    def parseMatrix(contentMatrix: str) -> list[list[float]]:
        matrix = []
        i = -1
        for line in contentMatrix.splitlines():
            i += 1
            if i == 0:
                m = re.match(r'		Matrix \d+', line)
                if m is not None:
                    continue
            if i == 1 and line == "		{":
                continue
            if 2 <= i <= 5:
                m = re.match(r'			([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)', line)
                if m is not None:
                    row = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                    matrix.append(row)
                    continue
            if i == 6 and line == "		}":
                    continue
            if i == 7 and line == "":
                continue
            raise "Could not parse matrix. Error in line " + str(i + 1) + ":\n" + contentMatrix

        return matrix

    @staticmethod
    def parseChildFlagsItem(contentChildFlagsItem: str) -> (str, str):
        flags1 = flags2 = ""
        i = -1
        for line in contentChildFlagsItem.splitlines():
            i += 1
            if i == 0 and line == "		Item":
                continue
            if i == 1 and line == "		{":
                continue
            if i == 2:
                m = re.match(r'			Flags1 (.+)', line)
                if m is not None:
                    flags1 = m.group(1)
                    continue
            if i == 3:
                m = re.match(r'			Flags2 (.+)', line)
                if m is not None:
                    flags2 = m.group(1)
                    continue
            if i == 4 and line == "		}":
                continue
            if i == 5 and line == "":
                continue
            raise "Could not parse ChildFlags item. Error in line " + str(i + 1) + ":\n" + contentChildFlagsItem

        return flags1, flags2


    polygons: list[Any]
    materials: list[Material]
    margin: float
    vertices: list[list[float]]
    shrunk: Union[list[list[float]], None]
    flags1: str
    flags2: str

    def __init__(self, polygons: list[Any], materials: list[Material], margin: float, vertices: list[list[float]], shrunk: Union[list[list[float]], None], flags1: str, flags2: str):
        self.polygons = polygons
        self.materials = materials
        self.margin = margin
        self.vertices = vertices
        self.shrunk = shrunk
        self.flags1 = flags1
        self.flags2 = flags2

    def transform(self, rotationQuaternion: list[float], scale: list[float], translation: list[float]) -> None:
        minScale = min(scale)
        for i in range(len(self.polygons)):
            self.polygons[i].scale(minScale)

        for i in range(len(self.vertices)):
            self.vertices[i] = Util.applyTransformation(self.vertices[i], rotationQuaternion, scale, translation)

        if self.shrunk is not None:
            for i in range(len(self.shrunk)):
                self.shrunk[i] = Util.applyTransformation(self.shrunk[i], rotationQuaternion, scale, translation)

    def isMergable(self, bound: "BoundBVH") -> bool:
        if self.getType() != bound.getType():
            return False
        elif self.margin != bound.margin:
            return False
        elif self.flags1 != bound.flags1:
            return False
        elif self.flags2 != bound.flags2:
            return False
        else:
            return True

    def merge(self, bound: "BoundBVH") -> None:
        if self.getType() != bound.getType():
            raise Exception("Cannot merge because types do not match")
        if self.margin != bound.margin:
            raise Exception("Cannot merge because margins do not match")
        if self.flags1 != bound.flags1:
            raise Exception("Cannot merge because flags1 do not match")
        if self.flags2 != bound.flags2:
            raise Exception("Cannot merge because flags2 do not match")

        materialsMapping = self.mergeMaterials(bound)

        vertexIndexOffset = len(self.vertices)
        for i in range(len(bound.polygons)):
            polygon = bound.polygons[i]
            polygon.offsetVertexIndex(vertexIndexOffset)
            # fix MaterialIndex according to materialsMapping
            polygon.materialIndex = materialsMapping[polygon.materialIndex]

        self.polygons.extend(bound.polygons)
        self.vertices.extend(bound.vertices)
        if self.shrunk is not None:
            self.shrunk.extend(bound.shrunk)

    def mergeMaterials(self, bound: "BoundBVH") -> list[int]:
        # compute materialsMapping to avoid redundant materials
        materialsMapping = []
        for i in range(len(bound.materials)):
            index = self.findIndexOfSameMaterial(bound.materials[i])
            materialsMapping.append(index)

        # append non-existing materials and update materialsMapping accordingly
        for i in range(len(materialsMapping)):
            if materialsMapping[i] < 0:
                self.materials.append(bound.materials[i])
                materialsMapping[i] = len(self.materials) - 1

        return materialsMapping

    def findIndexOfSameMaterial(self, material: Material):
        for i in range(len(self.materials)):
            if self.materials[i].equals(material):
                return i
        return -1

    def getType(self) -> str:
        if self.shrunk is None:
            return "BoundBVH"
        else:
            return "BoundGeometry"

    def writePhBound(self, file: IO):
        bbox = self.computeBoundingBox()
        bsphere = bbox.getEnclosingSphere()

        file.write("""		phBound
		{
			Type """ + self.getType() + """
			Radius """ + Util.floatToStr(bsphere.radius) + """
			AABBMax """ + Util.vertexToStr(bbox.max) + """
			AABBMin """ + Util.vertexToStr(bbox.min) + """
			Centroid """ + Util.vertexToStr(bsphere.center) + """
			CG """ + Util.vertexToStr(bsphere.center) + """ 
""")

        self.writePolygons(file)

        file.write("			GeometryCenter " + Util.vertexToStr(bsphere.center) + "\n")

        self.writeVertices(file, bsphere.center)
        file.write("			VertexColors null\n")

        self.writeMaterials(file)
        file.write("			MaterialColors null\n")

        file.write("			Margin " + Util.floatToStr(self.margin) + "\n")

        self.writeShrunk(file, bsphere.center)

        file.write("		}\n")

    def writeChildTransformsMatrix(self, file: IO, index: int):
        file.write("""		Matrix """ + str(index) + """
		{
			1.00000000 0.00000000 0.00000000
			0.00000000 1.00000000 0.00000000
			0.00000000 0.00000000 1.00000000
			0.00000000 0.00000000 0.00000000
		}
""")

    def writeChildFlagItem(self, file: IO):
        file.write("""		Item
		{
			Flags1 """ + self.flags1 + """
			Flags2 """ + self.flags2 + """
		}
""")

    def writePolygons(self, file: IO):
        numPolygons = len(self.polygons)
        file.write("			Polygons " + str(numPolygons) + """
			{
""")
        for i in range(numPolygons):
            file.write(self.polygons[i].asPolygonString(i))
        file.write("			}\n")

    def writeVertices(self, file: IO, geometryCenter: list[float]):
        numVertices = len(self.vertices)
        file.write("			Vertices " + str(numVertices) + """
			{
""")
        self.writeVertexList(file, self.vertices, geometryCenter)
        file.write("			}\n")

    def writeShrunk(self, file: IO, geometryCenter: list[float]):
        if self.shrunk is None:
            return

        numShrunk = len(self.shrunk)
        file.write("			Shrunk " + str(numShrunk) + """
			{
""")
        self.writeVertexList(file, self.shrunk, geometryCenter)
        file.write("			}\n")

    def writeVertexList(self, file: IO, vertices: list[list[float]], geometryCenter: list[float]):
        for i in range(len(vertices)):
            vertex = np.subtract(self.vertices[i], geometryCenter)
            file.write("				" + Util.vertexToStr(vertex) + "\n")

    def writeMaterials(self, file: IO):
        numMaterials = len(self.materials)
        file.write("			Materials " + str(numMaterials) + """
			{
""")
        for i in range(numMaterials):
            file.write(self.materials[i].asMaterialString(i))
        file.write("			}\n")

    def computeBoundingBox(self) -> BBox:
        # TODO store BBox to avoid duplicate computation when called mutiple times
        bbox = BBox.createReversedInfinityBox()
        for i in range(len(self.polygons)):
            self.polygons[i].extendBoundingBox(bbox, self.vertices)

        return bbox
