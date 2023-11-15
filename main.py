import distutils.util
import getopt
import os.path
import re
import shutil
import sys
import json

from matplotlib import pyplot

from worker.EntropyCreator import EntropyCreator
from worker.reducer.Reducer import Reducer
from worker.vegetation_creator.VegetationCreator import VegetationCreator
from worker.clustering.Clustering import Clustering
from worker.lod_map_creator.LodMapCreator import LodMapCreator
from worker.sanitizer.Sanitizer import Sanitizer
from worker.static_col_creator.StaticCollisionCreator import StaticCollisionCreator
from worker.statistics.StatisticsPrinter import StatisticsPrinter

PATTERN_MAP_NAME = "[a-z][a-z0-9_]*[a-z0-9]"


def moveDirectory(src: str, dest: str):
    for filename in os.listdir(src):
        shutil.move(os.path.join(src, filename), dest)


def copyDirectory(src: str, dest: str):
    for filename in os.listdir(src):
        shutil.copy(os.path.join(src, filename), dest)


def main(argv):
    inputDir = None
    outputDir = None
    vegetationCreator = False
    clustering = False
    numClusters = -1
    polygon = None
    reducer = False
    reducerResolution = 30.0
    reducerAdaptScaling = False
    clusteringPrefix = None
    clusteringExcluded = None
    staticCol = False
    lodMap = False
    clearLod = False
    sanitizer = False
    entropy = False
    statistics = False
    prefix = None

    usageMsg = "main.py --inputDir <input directory> --outputDir <output directory> --prefix=<PREFIX> " \
               "--reducer=<on|off> --reducerResolution=<float (default 30)> --reducerAdaptScaling=<on|off>" \
               "--clustering=<on|off> --numClusters=<integer> --polygon=<list of x,y coordinates in CCW order> " \
               "--clusteringPrefix=<CLUSTERING_PREFIX> --clusteringExcluded=<comma-separated list of ymaps to exclude> " \
               "--entropy=<on|off> --sanitizer=<on|off> --staticCol=<on|off> --lodMap=<on|off> --statistics=<on|off>"

    try:
        opts, args = getopt.getopt(argv, "h?i:o:",
            ["help", "inputDir=", "outputDir=", "reducer=", "reducerResolution=", "reducerAdaptScaling=",
                "clustering=", "numClusters=", "polygon=", "clusteringPrefix=", "clusteringExcluded=",
                "staticCol=", "prefix=", "lodMap=", "clearLod=", "sanitizer=", "entropy=", "statistics=", "vegetationCreator="])
    except getopt.GetoptError:
        print("ERROR: Unknown argument. Please see below for usage.")
        print(usageMsg)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(usageMsg)
            sys.exit(0)
        elif opt in ("-i", "--inputDir"):
            inputDir = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        elif opt == "--prefix":
            prefix = arg
        elif opt == "--vegetationCreator":
            vegetationCreator = bool(distutils.util.strtobool(arg))
        elif opt == "--reducer":
            reducer = bool(distutils.util.strtobool(arg))
        elif opt == "--reducerResolution":
            reducerResolution = float(arg)
            if reducerResolution <= 0:
                print("ERROR: reducerResolution must be positive")
                sys.exit(2)
        elif opt == "--reducerAdaptScaling":
            reducerAdaptScaling = bool(distutils.util.strtobool(arg))
        elif opt == "--clustering":
            clustering = bool(distutils.util.strtobool(arg))
        elif opt == "--clusteringPrefix":
            clusteringPrefix = arg
        elif opt == "--clusteringExcluded":
            clusteringExcluded = list(map(str.strip, arg.split(',')))
        elif opt == "--numClusters":
            numClusters = int(arg)
        elif opt == "--polygon":
            polygon = json.loads(arg)
        elif opt == "--staticCol":
            staticCol = bool(distutils.util.strtobool(arg))
        elif opt == "--lodMap":
            lodMap = bool(distutils.util.strtobool(arg))
        elif opt == "--clearLod":
            clearLod = bool(distutils.util.strtobool(arg))
        elif opt == "--sanitizer":
            sanitizer = bool(distutils.util.strtobool(arg))
        elif opt == "--entropy":
            entropy = bool(distutils.util.strtobool(arg))
        elif opt == "--statistics":
            statistics = bool(distutils.util.strtobool(arg))

    if not vegetationCreator and not reducer and not clustering and not staticCol and not lodMap and not sanitizer and not entropy and not statistics:
        print("ERROR: No goal specified, nothing to do.")
        print(usageMsg)
        sys.exit(2)

    if not prefix:
        prefix = input("Prefix of this project?")

    if not re.match(PATTERN_MAP_NAME, prefix):
        print("ERROR: prefix must contain only a-z 0-9 _ and must start with a letter and must not end in _")
        sys.exit(2)

    if clusteringPrefix is not None and not re.match(PATTERN_MAP_NAME, clusteringPrefix):
        print("ERROR: clusteringPrefix must contain only a-z 0-9 _ and must start with a letter and must not end in _")
        sys.exit(2)

    if not inputDir:
        inputDir = input("Input directory (containing the .ymap.xml files)?")
    inputDir = os.path.abspath(inputDir)

    if not os.path.isdir(inputDir):
        print("ERROR: inputDir " + inputDir + " is not a directory")
        sys.exit(2)

    if not outputDir:
        outputDir = os.path.join(inputDir, "generated")
    outputDir = os.path.abspath(outputDir)

    if os.path.exists(outputDir):
        if not os.path.isdir(outputDir):
            print("ERROR: outputDir " + outputDir + " is not a directory")
            sys.exit(2)

        print("outputDir " + outputDir + " already exists.")
        clearDirConfirmation = input("Are you sure you want to clear directory " + outputDir +
                                     "?\nWARNING: This irreversibly erases all files within that directory!\nPlease enter yes or no: ")
        # this if-statement is very important to prevent unintended deletion of a directory
        if clearDirConfirmation == "yes" or clearDirConfirmation == "y":
            shutil.rmtree(outputDir)
        else:
            sys.exit(0)

    nextInputDir = inputDir

    os.makedirs(outputDir)

    tempOutputDir = os.path.join(outputDir, "_temp_")
    os.makedirs(tempOutputDir)

    if vegetationCreator:
        vegetationCreatorWorker = VegetationCreator(nextInputDir, os.path.join(tempOutputDir, "vegetationCreator"), prefix)
        vegetationCreatorWorker.run()

        nextInputDir = vegetationCreatorWorker.outputDir

    if entropy:
        entropyCreator = EntropyCreator(nextInputDir, os.path.join(tempOutputDir, "entropy"), False, True, False, True)
        entropyCreator.run()

        nextInputDir = entropyCreator.outputDir

    if reducer:
        reducerWorker = Reducer(nextInputDir, os.path.join(tempOutputDir, "reducer"), prefix, reducerResolution, reducerAdaptScaling)
        reducerWorker.run()

        nextInputDir = reducerWorker.outputDir

    if clustering:
        clusteringWorker = Clustering(nextInputDir, os.path.join(tempOutputDir, "clustering"), prefix,
            numClusters, polygon, clusteringPrefix, clusteringExcluded)
        clusteringWorker.run()

        nextInputDir = clusteringWorker.outputDir

    if sanitizer:
        sanitizerWorker = Sanitizer(nextInputDir, os.path.join(tempOutputDir, "sanitizer"))
        sanitizerWorker.run()

        nextInputDir = sanitizerWorker.outputDir

    if lodMap or clearLod:
        lodMapCreator = LodMapCreator(nextInputDir, os.path.join(tempOutputDir, "lod_map"), prefix, clearLod)
        lodMapCreator.run()

        outputSlodDir = os.path.join(outputDir, prefix + "_slod")
        os.makedirs(outputSlodDir)
        moveDirectory(lodMapCreator.getOutputDirModels(), outputSlodDir)

        outputMeshesDir = os.path.join(outputDir, os.path.basename(lodMapCreator.getOutputDirMeshes()))
        os.makedirs(outputMeshesDir)
        moveDirectory(lodMapCreator.getOutputDirMeshes(), outputMeshesDir)

        nextInputDir = lodMapCreator.getOutputDirMaps()

    if staticCol:
        staticCollisionCreator = StaticCollisionCreator(nextInputDir, os.path.join(tempOutputDir, "static_col"))
        staticCollisionCreator.run()

        outputStaticColsDir = os.path.join(outputDir, prefix)
        os.makedirs(outputStaticColsDir)
        moveDirectory(staticCollisionCreator.getOutputDirCollisionModels(), outputStaticColsDir)

        nextInputDir = staticCollisionCreator.getOutputDirMaps()

    if statistics:
        statisticsPrinter = StatisticsPrinter(nextInputDir)
        statisticsPrinter.run()

    outputMapsDir = os.path.join(outputDir, prefix + "_metadata")
    os.makedirs(outputMapsDir)
    if not os.path.samefile(nextInputDir, inputDir):
        moveDirectory(nextInputDir, outputMapsDir)
    # else:
    #    no need to duplicate the input dir
    #    copyDirectory(nextInputDir, outputDir + "/maps")
    shutil.rmtree(tempOutputDir)

    pyplot.show(block=True)


if __name__ == "__main__":
    main(sys.argv[1:])
