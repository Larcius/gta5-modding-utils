# gta5-modding-utils

## Prerequisites

### Python and Miniconda

Since these scripts are written in Python you need to install Python in order to run these scripts.
I recommend using Miniconda for an easy installation of the required package.
https://docs.conda.io/en/latest/miniconda.html

Once you installed Miniconda you need to create a new environment and install all required python packages:
```commandline
conda create --name gta5-modding-utils -y
conda install --name gta5-modding-utils -c conda-forge scikit-learn -y
conda install --name gta5-modding-utils -c conda-forge transforms3d -y
conda install --name gta5-modding-utils -c conda-forge natsort -y
conda env config vars set OMP_NUM_THREADS=1 --name gta5-modding-utils
```

Now your Pyhton environment is ready to use:
````commandline
conda activate gta5-modding-utils
````

### Resources from GTA 5

For legal reasons I cannot ship original GTA 5 files, so you need to provide them as described here.

#### YTYP

You need to provide the .ytyp files from your GTA 5 installation so that the required information
(e.g. bounding box of an entity) is available to the scripts.
To do so please use OpenIV and export all needed ytyp files in openFormats
(right click on the file and then "Export to META/XML (.xml)")
to the directory `ytyp\resources`
For example, using OpenIV goto `<Grand Theft Auto V>\x64i.rpf\levels\gta5\props\vegetation\v_trees.rpf\` and extract `v_trees.ytyp`

If you don't know what .ytyp files you need then just run the script and it will tell you which archetype is missing/unknown.
For those use OpenIV search ("Tools" -> "Search") to find the location of the .ydr file and within this directory there is the .ytyp file

#### YDR

If you want to create static collision models then you need to extract the models (.ydr files) that then should be part of the
static collision model into the directory `models\resources`
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
python main.py --inputDir="<DIRECTORY CONTAINING THE .ymap.xml files>" --prefix="<PROJECT_PREFIX>" --clustering=on
````
For example if you want to run sanitizing, static collision model creator, lod model creator and lod/slod map creator use this command:
````commandline
python main.py --inputDir="<DIRECTORY CONTAINING THE .ymap.xml files>" --prefix="<PROJECT_PREFIX>" --lodModel=on --sanitizer=on --staticCol=on --lodMap=on
````

After that you will see the output in the given directory (if not explicitly stated then it's in a subdirectory `generated` in the provided input directory).
Finally, you need to import these files in your dlc.rpf (please have a look at GTA V Remastered: Enhanced for an example structure).
OpenIV automatically converts these openFormats files back to binary files on importing.

Note: The directories `_lod_meshes` and `_slod_meshes` are used when importing the files from directory `<PREFIX>_lod` and `<PREFIX>_slod`. 


## Used Python packages

* https://github.com/scikit-learn/scikit-learn
* https://github.com/matthew-brett/transforms3d
* https://github.com/SethMMorton/natsort
