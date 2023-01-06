import re


class Material:
    materialIndex: int
    polyFlags: str

    @staticmethod
    def parse(content: str) -> "Material":
        materialIndex = -1
        polyFlags = ""
        i = -1
        for line in content.splitlines():
            i += 1
            if i == 0:
                m = re.match(r'				Material \d+$', line)
                if m is not None:
                    continue
            elif i == 1 and line == "				{":
                continue
            elif i == 2:
                m = re.match(r'					MaterialIndex (\d+)$', line)
                if m is not None:
                    materialIndex = int(m.group(1))
                    continue
            elif i == 3 and line == "					ProcId 0":
                continue
            elif i == 4 and line == "					RoomId 0":
                continue
            elif i == 5 and line == "					PedDensity 0":
                continue
            elif i == 6:
                m = re.match(r'					PolyFlags (.+)$', line)
                if m is not None:
                    polyFlags = m.group(1)
                    continue
            elif i == 7 and line == "					MaterialColorIndex 0":
                continue
            elif i == 8 and line == "				}":
                continue
            elif i == 9 and line == "":
                continue

            raise Exception("Could not parse Material. Error in line " + str(i + 1) + ":\n" + content)

        return Material(materialIndex, polyFlags)

    def __init__(self, materialIndex: int, polyFlags: str):
        self.materialIndex = materialIndex
        self.polyFlags = polyFlags

    def equals(self, other: "Material") -> bool:
        return self.materialIndex == other.materialIndex and self.polyFlags == other.polyFlags

    def asMaterialString(self, materialIndex: int) -> str:
        return """				Material """ + str(materialIndex) + """
				{
					MaterialIndex """ + str(self.materialIndex) + """
					ProcId 0
					RoomId 0
					PedDensity 0
					PolyFlags """ + self.polyFlags + """
					MaterialColorIndex 0
				}
"""
