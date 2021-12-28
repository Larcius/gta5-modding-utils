# TODO refactoring needed

import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from matplotlib import pyplot
import math
import os
import re
import shutil

NUM_CLUSTERS = 25
MAP_NAME = "vremastered"

generatedDir = os.path.join(os.path.dirname(__file__), "generated")
if os.path.exists(generatedDir):
    shutil.rmtree(generatedDir)
os.mkdir(generatedDir)

f = open("default_post_entities.xml", 'r')
defaultPostEntities = f.read()
f.close()

numDigitsMapIndices = math.ceil(math.log(NUM_CLUSTERS, 10))
coords = []
customPostEntities = []
for filename in os.listdir("maps"):
    if not filename.endswith(".ymap.xml"):
        continue

    f = open(os.path.join("maps", filename), 'r')
    content = f.read()
    f.close()

    indexEntitiesStart = content.index("  <entities>")
    indexEntitiesEnd = content.index("  </entities>")

    contentPreEntities = content[:indexEntitiesStart + 13]
    contentPostEntities = content[indexEntitiesEnd:]

    if not contentPostEntities.startswith(defaultPostEntities):
        customPostEntities.append(filename)

    pattern = re.compile(
        '[\t ]*<Item type="CEntityDef">' +
        '\\s*<archetypeName>[^<]+</archetypeName>' +
        '(?:\\s*<[^/].*>)*' +
        '\\s*<position x="([^"]+)" y="([^"]+)" z="[^"]+"\\s*/>' +
        '(?:\\s*<[^/].*>)*' +
        '\\s*</Item>[\r\n]+'
    )

    for matchobj in re.finditer(pattern, content):
        coords.append([float(matchobj.group(1)), float(matchobj.group(2))])

# define dataset
X = np.array(coords)

# define the model
model = KMeans(n_clusters=NUM_CLUSTERS, n_init=200, max_iter=5000, algorithm="full")

if hasattr(model, "fit_predict"):
    # fit the model and assign a cluster to each example
    yhat = model.fit_predict(X)
else:
    # fit the model
    model.fit(X)

    # assign a cluster to each example
    yhat = model.predict(X)

clusters = unique(yhat)
outputFiles = {}
for cluster in clusters:
    outputFiles[cluster] = open(os.path.join("generated", MAP_NAME + "_" + str(cluster + 1).zfill(numDigitsMapIndices) + ".ymap.xml"), 'w')
    outputFiles[cluster].write(contentPreEntities)

i = 0
for filename in os.listdir("maps"):
    if not filename.endswith(".ymap.xml"):
        continue

    f = open(os.path.join("maps", filename), 'r')
    content = f.read()
    f.close()

    for matchobj in re.finditer(pattern, content):
        cluster = yhat[i]
        outputFiles[cluster].write(matchobj.group(0))
        i += 1

for cluster in clusters:
    outputFiles[cluster].write(contentPostEntities)
    outputFiles[cluster].close()

for custom in customPostEntities:
    print("custom content after </entities> in file " + custom)

# further is only needed to immediately show clustering result

# create scatter plot for samples from each cluster
N = len(clusters)
cmap = pyplot.cm.get_cmap("gist_ncar", N + 1)
i = 0
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)

    print(len(row_ix[0]))

    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], color=cmap(i))

    i += 1

pyplot.gca().set_aspect('equal')

# show the plot
pyplot.show()
