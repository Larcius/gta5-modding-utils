# Fixing z coordinates

This python script z-fixer.py in combination with the C# script HeightMapGenerator.3.cs (requires ScriptHookVDotNet v3)
fixes z coordinates of trees, palms and bushes.

Please perform the following steps:
1. put your .ymap.xml files in this directory: `gta5-modding-utils\z_fixer\maps`
2. open the console and change directory to `gta5-modding-utils\z_fixer`
3. run this command: ```python z-fixer.py --extract``` (so this creates a new file `gta5-modding-utils\z_fixer\generates\coords.txt`)
4. copy the file `gta5-modding-utils\z_fixer\scripts\HeightMapGenerator.3.cs` to `<Grand Theft Auto V game folder>\scripts`
5. start the game (but without using these ymap files because otherwise the z coordinates are incorrect due to these ymaps)
6. in story mode press `F10` and type in the absolute path to the `hmap.txt` and press enter (your character will jump around the map and the heights are recorded; if you want to abort just press `F10` again)
7. as soon as your character stops jumping around you are done and there is a new file `<Grand Theft Auto V game folder>\hmap.txt`
8. move that file to `gta5-modding-utils\z_fixer\heights\`
9. now run this command: ```python z-fixer.py --fix```
10. in `gta5-modding-utils\z_fixer\generated` there are now your fixed ymaps.