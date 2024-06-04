# gta5-modding-utils

## Prerequisites

### Python and Miniconda

Since these scripts are written in Python you need to install Python in order to run these scripts.
I recommend using Miniconda for an easy installation of the required package.
https://docs.conda.io/en/latest/miniconda.html

Once you installed Miniconda you need to create a new environment and install all required python packages:
```commandline
conda env create -y -f environment.yml
```

Now your environment is ready to use and from now on all you need to do is to activate this environment:
````commandline
conda activate gta5-modding-utils-env
````

### Resources from GTA 5

For legal reasons I cannot ship original GTA 5 files, so you need to provide them as described here.

#### YTYP

You need to provide the .ytyp files from your GTA 5 installation so that the required information
(e.g. bounding box of an entity) is available to the scripts.
To do so please use OpenIV and export all needed ytyp files in openFormats
(right click on the file and then "Export to META/XML (.xml)")
to the directory `resources\ytyp`
For example, using OpenIV goto `<Grand Theft Auto V>\x64i.rpf\levels\gta5\props\vegetation\v_trees.rpf\` and extract `v_trees.ytyp`

If you don't know what .ytyp files you need then just run the script and it will tell you which archetype is missing/unknown.
For those use OpenIV search ("Tools" -> "Search") to find the location of the .ydr file and within this directory there is the .ytyp file

#### YDR

If you want to create static collision models then you need to extract the models (.ydr files) that then should be part of the
static collision model into the directory `resources\models`
(again extract it in openFormats by right-clicking and then "Exports to openFormats (.odr)")
For example, using OpenIV goto `<Grand Theft Auto V>\x64i.rpf\levels\gta5\props\vegetation\v_trees.rpf\` and extract `prop_tree_cedar_02.ydr`

If you don't know where a specific model is located then again please use OpenIV to search for it.


## Running the scripts

The scripts expect an input directory where your map files are located (again in openFormats, so they have to end in .ymap.xml)
If you currently only have them as binary files (ending in .ymap) please use OpenIV and then "File" -> "Open folder" and select your folder.
Then select all of your ymap files and export them in openFormats (right click and then "Export to META/XML (.xml)").
I recommend exporting them into another subdirectory so that the directory you state as input directory only contains these .ymap.xml files.

Now that you have your Python environment prepared and provided the necessary files you can finally apply the scripts.
For example if you want to perform clustering (read entities from all ymaps, perform clustering and create new ymaps each containing one cluster) use this command:
````commandline
python main.py --inputDir="<DIRECTORY CONTAINING THE .ymap.xml files>" --prefix="<PROJECT_PREFIX>" --clustering=on --clusteringPrefix="<CLUSTERING_PREFIX>"
````
This will automatically detect the number of clusters so that the extends of each map is reasonable.
However, if you want a specific number here please add `--numClusters=<NUMBER>` to the command.
But please keep in mind that having too large map extends may have a huge impact on performance and stability, so only use a specific value here if you are aware of that.

For example if you want to run sanitizing, clustering, static collision model creator, lod model creator and lod/slod map creator use this command:
````commandline
python main.py --inputDir="<DIRECTORY CONTAINING THE .ymap.xml files>" --prefix="<PROJECT_PREFIX>" --sanitizer=on --clustering=on --clusteringPrefix="<CLUSTERING_PREFIX>" --staticCol=on --lodMap=on
````

After that you will see the output in the given directory (if not explicitly stated then it's in a subdirectory `generated` in the provided input directory).
Finally, you need to import these files in your dlc.rpf (please have a look at GTA V Remastered: Enhanced for an example structure).
OpenIV automatically converts these openFormats files back to binary files on importing.

Note: The directory `_slod_meshes` is used when importing the files from directory `<PREFIX>_slod`.


## Video tutorial

Thanks to Xotiic for making a tutorial video:
https://www.youtube.com/watch?v=KDduBaZEdeQ


## Used Python packages

* https://github.com/scikit-learn/scikit-learn
* https://github.com/matthew-brett/transforms3d
* https://github.com/SethMMorton/natsort
* https://github.com/matplotlib/matplotlib
