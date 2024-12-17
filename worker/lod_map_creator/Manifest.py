import os
import re

from natsort import natsorted

from common.Util import Util
from common.ytyp.YtypItem import YtypItem


class Manifest:
    ytypItems: dict[str, YtypItem]
    imapsToYtyps: dict[str, set[str]]
    mapsDir: str
    metadataDir: str

    def __init__(self, ytypItems: dict[str, YtypItem], mapsDir: str, metadataDir: str):
        self.ytypItems = ytypItems
        self.mapsDir = mapsDir
        self.metadataDir = metadataDir
        self.imapsToYtyps = {}

    def parseYmaps(self):
        for filename in natsorted(os.listdir(self.mapsDir)):
            if not filename.endswith(".ymap.xml"):
                continue

            file = open(os.path.join(self.mapsDir, filename), 'r')
            content = file.read()
            file.close()

            self.parseYmapContent(Util.getMapnameFromFilename(filename), content)

    def parseYmapContent(self, mapName: str, content: str):
        ytyps = set()

        for match in re.finditer('<Item type="CEntityDef">\\s*<archetypeName>([^<]+)</archetypeName>', content):
            archetypeName = match.group(1).lower()

            if archetypeName not in self.ytypItems:
                print("WARNING: could not find archetype " + archetypeName + ". Proceeding without it but that means it is missing in manifest.")
                continue

            ytyp = self.ytypItems[archetypeName].parent
            ytyps.add(ytyp)

        self.imapsToYtyps[mapName] = ytyps

    def writeManifest(self):
        file = open(os.path.join(self.metadataDir, '_manifest.ymf.xml'), 'w')
        file.write("""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<CPackFileMetaData>
	<MapDataGroups/>
	<HDTxdBindingArray/>
	<imapDependencies/>
	<imapDependencies_2>
""")

        for mapName in natsorted(self.imapsToYtyps):
            file.write("""		<Item>
			<imapName>""" + mapName + """</imapName>
			<manifestFlags/>
			<itypDepArray>
""")

            for ytyp in natsorted(self.imapsToYtyps[mapName]):
                file.write("				<Item>" + ytyp + "</Item>\n")

            file.write("""			</itypDepArray>
		</Item>
""")

        file.write("""	</imapDependencies_2>
	<itypDependencies_2/>
	<Interiors/>
</CPackFileMetaData>""")

        file.close()
