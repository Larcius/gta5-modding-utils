import re
from typing import IO

from common.BoundingGeometry import BoundingGeometry
from worker.static_col_creator.BoundBVH import BoundBVH

from common.Util import Util


class BoundComposite:

    DEFAULT_CHILD_FLAGS = """		Item
		{
			Flags1 MAP_DYNAMIC
			Flags2 OBJECT
		}
"""

    @staticmethod
    def parse(content: str) -> "BoundComposite":
        # modes:
        #  0: start of BoundComposite (before Children)
        #  1: Children
        #  2: within Children
        #  3: start of single phBound
        #  4: within single phBound
        #  5: after Children expecting ChildTransforms
        #  6: start of ChildTransforms
        #  7: within ChildTransforms
        #  8: start of single Matrix
        #  9: within single Matrix
        # 10: after ChildTransforms expecting ChildFlags
        # 11: start of ChildFlags
        # 12: within ChildFlags
        # 13: start of single Item
        # 14: within single Item
        # 15: after ChildFlags expecting end of BoundComposite
        # 16: end of BoundComposite expecting no more content (except empty line)
        mode = 0

        i = -1
        phBound = ""
        children = []
        matrix = ""
        childTransforms = []
        item = ""
        childFlags = []
        for line in content.splitlines():
            i += 1
            if i == 0 and line == "Version 43 31":
                continue
            elif i == 1 and line == "{":
                continue
            elif i == 2:
                m = re.match(r'	Type BoundComposite$', line)
                if m is not None:
                    continue
            elif i == 3:
                m = re.match(r'	Radius ([+-]?\d+\.\d+)$', line)
                if m is not None:
                    continue
            elif 4 <= i <= 7:
                m = re.match(r'	(?:AABBMax|AABBMin|Centroid|CG) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)$', line)
                if m is not None:
                    continue

            # Children
            elif i == 8:
                m = re.match(r'	Children \d+$', line)
                if m is not None:
                    mode = 1
                    continue
            elif mode == 1:
                if line == "	{":
                    mode = 2
                    continue
            elif mode == 2:
                if line == "		phBound":
                    phBound = line + "\n"
                    mode = 3
                    continue
                elif line == "	}":
                    mode = 5
                    continue
            elif mode == 3:
                if line == "		{":
                    phBound += line + "\n"
                    mode = 4
                    continue
            elif mode == 4:
                if line.startswith("			"):
                    phBound += line + "\n"
                    continue
                elif line == "		}":
                    phBound += line + "\n"
                    children.append(phBound)
                    mode = 2
                    continue

            # ChildTransforms
            elif mode == 5:
                m = re.match(r'	ChildTransforms \d+$', line)
                if m is not None:
                    mode = 6
                    continue
            elif mode == 6:
                if line == "	{":
                    mode = 7
                    continue
            elif mode == 7:
                m = re.match(r'		Matrix \d+$', line)
                if m is not None:
                    matrix = line + "\n"
                    mode = 8
                    continue
                elif line == "	}":
                    mode = 10
                    continue
            elif mode == 8:
                if line == "		{":
                    matrix += line + "\n"
                    mode = 9
                    continue
            elif mode == 9:
                if line.startswith("			"):
                    matrix += line + "\n"
                    continue
                elif line == "		}":
                    matrix += line + "\n"
                    childTransforms.append(matrix)
                    mode = 7
                    continue

            # ChildFlags
            elif mode == 10:
                m = re.match(r'	ChildFlags \d+$', line)
                if m is not None:
                    mode = 11
                    continue

                m = re.match(r'	ChildFlags null$', line)
                if m is not None:
                    childFlags = [BoundComposite.DEFAULT_CHILD_FLAGS] * len(childTransforms)
                    mode = 15
                    continue
            elif mode == 11:
                if line == "	{":
                    mode = 12
                    continue
            elif mode == 12:
                if line == "		Item":
                    item = line + "\n"
                    mode = 13
                    continue
                elif line == "	}":
                    mode = 15
                    continue
            elif mode == 13:
                if line == "		{":
                    item += line + "\n"
                    mode = 14
                    continue
            elif mode == 14:
                if line.startswith("			"):
                    item += line + "\n"
                    continue
                elif line == "		}":
                    item += line + "\n"
                    childFlags.append(item)
                    mode = 12
                    continue

            # end of BoundComposite
            if mode == 15 and line == "}":
                mode = 16
                continue
            elif mode == 16 and line == "":
                continue

            raise Exception("Could not parse BoundComposite. Error in line " + str(i + 1) + ":\n" + content)

        assert len(children) == len(childTransforms) == len(childFlags)

        boundBVHs = []
        for i in range(len(children)):
            boundBVH = BoundBVH.parse(children[i], childTransforms[i], childFlags[i])
            boundBVHs.append(boundBVH)

        return BoundComposite(boundBVHs)


    children: list[BoundBVH]

    def __init__(self, children: list[BoundBVH]):
        self.children = children

    def transform(self, rotationQuaternion: list[float], scale: list[float], translation: list[float]) -> None:
        for i in range(len(self.children)):
            self.children[i].transform(rotationQuaternion, scale, translation)

    def merge(self, boundComposite: "BoundComposite") -> None:
        # TODO improve since this is O(n^2)
        for child in boundComposite.children:
            merged = False
            for candidate in self.children:
                if candidate.isMergable(child):
                    candidate.merge(child)
                    merged = True
                    break
            if not merged:
                self.children.append(child)

    def splitIntoDefaultMaHi(self) -> ("BoundComposite", "BoundComposite", "BoundComposite"):
        childrenHi = []
        childrenMa = []
        childrenDefault = []
        for child in self.children:
            if child.flags1 == "MAP_WEAPON":
                childrenHi.append(child)
            elif child.flags1 == "FORMATS_EMPTY_FLAGS":
                childrenMa.append(child)
            else:
                childrenDefault.append(child)

        return BoundComposite(childrenDefault), BoundComposite(childrenMa), BoundComposite(childrenHi)

    def writeToFile(self, path: str):
        if len(self.children) == 0:
            return

        file = open(path, 'w')
        self.writeHeader(file)
        self.writeChildren(file)
        self.writeChildTransforms(file)
        self.writeChildFlags(file)
        file.write("}\n")
        file.close()

    def writeHeader(self, file: IO):
        boundingGeometry = self.computeBoundingGeometry()
        bbox = boundingGeometry.getBox()
        bsphere = boundingGeometry.getSphere()
        file.write("""Version 43 31
{
	Type BoundComposite
	Radius """ + Util.floatToStr(bsphere.radius) + """
	AABBMax """ + Util.vectorToStr(bbox.max) + """
	AABBMin """ + Util.vectorToStr(bbox.min) + """
	Centroid """ + Util.vectorToStr(bsphere.center) + """
	CG """ + Util.vectorToStr(bsphere.center) + """ 
""")

    def writeChildren(self, file: IO):
        numChildren = len(self.children)
        file.write("	Children " + str(numChildren) + "\n	{\n")
        for i in range(numChildren):
            self.children[i].writePhBound(file)
        file.write("	}\n")

    def writeChildTransforms(self, file: IO):
        numChildren = len(self.children)
        file.write("	ChildTransforms " + str(numChildren) + "\n	{\n")
        for i in range(numChildren):
            self.children[i].writeChildTransformsMatrix(file, i)
        file.write("	}\n")

    def writeChildFlags(self, file: IO):
        numChildren = len(self.children)
        file.write("	ChildFlags " + str(numChildren) + "\n	{\n")
        for i in range(numChildren):
            self.children[i].writeChildFlagItem(file)
        file.write("	}\n")

    def computeBoundingGeometry(self) -> BoundingGeometry:
        boundingGeometry = BoundingGeometry()
        for i in range(len(self.children)):
            childBoundingGeometry = self.children[i].getBoundingGeometry()
            boundingGeometry.extendByBoundingGeometry(childBoundingGeometry)

        return boundingGeometry
