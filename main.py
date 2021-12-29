import distutils.util
import getopt
import os.path
import re
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
    inputDir = None
    outputDir = None
    staticCol = False
    lodMap = False
    sanitizer = False
    entropy = False
    statistics = False
    prefix = None

    usageMsg = "main.py --inputDir <input directory> --outputDir <output directory> --prefix=<PREFIX> --entropy=<on|off> --sanitizer=<on|off> " \
               "--staticCol=<on|off> --lodMap=<on|off> --statistics=<on|off> "

    try:
        opts, args = getopt.getopt(argv, "h?i:o:", ["help", "inputDir=", "outputDir=", "staticCol=", "prefix=", "lodMap=", "sanitizer=", "entropy=", "statistics="])
    except getopt.GetoptError:
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

    if not staticCol and not lodMap and not sanitizer and not entropy and not statistics:
        print("No goal specified, nothing to do.")
        print(usageMsg)
        sys.exit(2)

    if not prefix:
        prefix = input("Prefix of this project?")

    if not re.match("[a-z][a-z0-9_]*[a-z0-9]", prefix):
        print("prefix must contain only a-z 0-9 _ and must start with a letter and must not end in _")
        sys.exit(2)

    if not inputDir:
        inputDir = input("Input directory (containing the .ymap.xml files)?")
    inputDir = os.path.abspath(inputDir)

    if not outputDir:
        outputDir = os.path.join(inputDir, "generated")
    outputDir = os.path.abspath(outputDir)

    if os.path.exists(outputDir):
        if not os.path.isdir(outputDir):
            raise ValueError("outputDir is not a directory")

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

        outputStaticColsDir = os.path.join(outputDir, prefix)
        os.makedirs(outputStaticColsDir)
        moveDirectory(staticCollisionCreator.getOutputDirCollisionModels(), outputStaticColsDir)

        nextInputDir = staticCollisionCreator.getOutputDirMaps()

    if lodMap:
        lodMapCreator = LodMapCreator(nextInputDir, os.path.join(tempOutputDir, "lod_map"), prefix)
        lodMapCreator.run()

        outputSlodDir = os.path.join(outputDir, prefix + "_slod")
        os.makedirs(outputSlodDir)
        moveDirectory(lodMapCreator.getOutputDirModels(), outputSlodDir)

        outputMeshesDir = os.path.join(outputDir, "_meshes")
        os.makedirs(outputMeshesDir)
        moveDirectory(lodMapCreator.getOutputDirMeshes(), outputMeshesDir)

        nextInputDir = lodMapCreator.getOutputDirMaps()

    if statistics:
        statisticsPrinter = StatisticsPrinter(nextInputDir)
        statisticsPrinter.run()

    outputMapsDir = os.path.join(outputDir, prefix + "_metadata")
    os.makedirs(outputMapsDir)
    if not os.path.samefile(nextInputDir, inputDir):
        moveDirectory(nextInputDir, outputMapsDir)
    #else:
    #    no need to duplicate the input dir
    #    copyDirectory(nextInputDir, outputDir + "/maps")
    shutil.rmtree(tempOutputDir)


if __name__ == "__main__":
    main(sys.argv[1:])
