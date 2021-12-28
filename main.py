import distutils.util
import getopt
import os.path
import shutil
import sys

from worker.EntropyCreator import EntropyCreator
from worker.lod_map_creator.LodMapCreator import LodMapCreator
from worker.sanitizer.Sanitizer import Sanitizer
from worker.static_col_creator.StaticCollisionCreator import StaticCollisionCreator
from worker.statistics.StatisticsPrinter import StatisticsPrinter


def moveDirectory(src: str, dest: str):
    for filename in os.listdir(src):
        shutil.move(os.path.join(src, filename), dest)


def copyDirectory(src: str, dest: str):
    for filename in os.listdir(src):
        shutil.copy(os.path.join(src, filename), dest)


def main(argv):
    inputDir = ''
    outputDir = None
    staticCol = False
    lodMap = False
    sanitizer = False
    entropy = False
    statistics = False
    prefix = "custommap"

    try:
        opts, args = getopt.getopt(argv, "h?i:o:", ["help", "inputDir=", "outputDir=", "staticCol=", "prefix=", "lodMap=", "sanitizer=", "entropy=", "statistics="])
    except getopt.GetoptError:
        print("main.py -i <inputfile> -o <outputDir> --prefix=<PREFIX> [--entropy=<False|True>] [--sanitizer=<False|True>] [--staticCol=<False|True>] [--lodMap=<False|True>] [--statistics=<False|True>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print("main.py -i <inputDir> -o <outputDir>")
            sys.exit()
        elif opt in ("-i", "--inputDir"):
            inputDir = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        elif opt == "--prefix":
            prefix = arg
        elif opt == "--staticCol":
            staticCol = bool(distutils.util.strtobool(arg))
        elif opt == "--lodMap":
            lodMap = bool(distutils.util.strtobool(arg))
        elif opt == "--sanitizer":
            sanitizer = bool(distutils.util.strtobool(arg))
        elif opt == "--entropy":
            entropy = bool(distutils.util.strtobool(arg))
        elif opt == "--statistics":
            statistics = bool(distutils.util.strtobool(arg))

    if not inputDir:
        inputDir = input("Input directory?")
    inputDir = os.path.abspath(inputDir)

    if not outputDir:
        outputDir = os.path.join(inputDir, "generated")
    outputDir = os.path.abspath(outputDir)

    if os.path.exists(outputDir):
        if not os.path.isdir(outputDir):
            raise ValueError("outputDir is not a directory")

        print("outputDir already exists")
        clearDirConfirmation = input("Are you sure you want to clear directory " + outputDir + "? (yes|no)\nWARNING: This irreversibly erases all files within that directory!")
        if not distutils.util.strtobool(clearDirConfirmation):
            # this statement is very important to prevent unintended deletion of a directory
            sys.exit(0)

    nextInputDir = inputDir

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.makedirs(outputDir)

    tempOutputDir = os.path.join(outputDir, "_temp_")
    os.makedirs(tempOutputDir)

    if entropy:
        entropyCreator = EntropyCreator(nextInputDir, os.path.join(tempOutputDir, "entropy"), False, True, True)
        entropyCreator.run()

        nextInputDir = entropyCreator.outputDir

    if sanitizer:
        sanitizerWorker = Sanitizer(nextInputDir, os.path.join(tempOutputDir, "sanitizer"))
        sanitizerWorker.run()

        nextInputDir = sanitizerWorker.outputDir

    if staticCol:
        staticCollisionCreator = StaticCollisionCreator(nextInputDir, os.path.join(tempOutputDir, "static_col"))
        staticCollisionCreator.run()

        outputStaticColsDir = os.path.join(outputDir, "static_col_models")
        os.makedirs(outputStaticColsDir)
        moveDirectory(staticCollisionCreator.getOutputDirCollisionModels(), outputStaticColsDir)

        nextInputDir = staticCollisionCreator.getOutputDirMaps()

    if lodMap:
        lodMapCreator = LodMapCreator(nextInputDir, os.path.join(tempOutputDir, "lod_map"), prefix)
        lodMapCreator.run()

        outputSlodDir = os.path.join(outputDir, "slod")
        os.makedirs(outputSlodDir)
        moveDirectory(lodMapCreator.getOutputDirModels(), outputSlodDir)

        outputMeshesDir = os.path.join(outputDir, "_meshes")
        os.makedirs(outputMeshesDir)
        moveDirectory(lodMapCreator.getOutputDirMeshes(), outputMeshesDir)

        nextInputDir = lodMapCreator.getOutputDirMaps()

    if statistics:
        statisticsPrinter = StatisticsPrinter(nextInputDir)
        statisticsPrinter.run()

    outputMapsDir = os.path.join(outputDir, "maps")
    os.makedirs(outputMapsDir)
    if nextInputDir == inputDir:
        dummy = ""
        # copyDirectory(nextInputDir, outputDir + "/maps")
    else:
        moveDirectory(nextInputDir, outputMapsDir)
    shutil.rmtree(tempOutputDir)


if __name__ == "__main__":
    main(sys.argv[1:])
