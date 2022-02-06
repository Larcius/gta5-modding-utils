# TODO refactoring needed

import os
import re
import random
import shutil

# using a specific seed to be able to get reproducible results
from natsort import natsorted

random.seed(a=0)

generatedDir = os.path.join(os.path.dirname(__file__), "generated")
if os.path.exists(generatedDir):
    shutil.rmtree(generatedDir)
os.mkdir(generatedDir)


def floatToStr(val):
    return "{:.8f}".format(val)


def repl(matchobj):
    scaleZ = float(matchobj.group(3))
    scaleXY = scaleZ * random.uniform(1, 1.1)

    scaleXY = round(scaleXY, 4)
    scaleZ = round(scaleZ, 4)

    return matchobj.group(1) + floatToStr(scaleXY) + matchobj.group(2)


for filename in natsorted(os.listdir(os.path.join(os.path.dirname(__file__), "maps"))):
    if not filename.endswith(".ymap.xml") or filename.endswith("_lod.ymap.xml"):
        continue

    f = open(os.path.join(os.path.dirname(__file__), "maps", filename), 'r')
    content = f.read()

    content_new = re.sub('(<Item type="CEntityDef">' +
                         '\\s*<archetypeName>prop_tree_pine_0[12]</archetypeName>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*<scaleXY\\s+value=")[^"]+("\\s*/>' +
                         '\\s*<scaleZ\\s+value="([^"]+)"\\s*/>' +
                         '(?:\\s*<[^/].*>)*' +
                         '\\s*</Item>)', repl, content, flags=re.M)

    f = open(os.path.join(os.path.dirname(__file__), "generated", filename), 'w')
    f.write(content_new)
