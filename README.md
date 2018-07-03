# Process-Srcmod-Files
Processing code for SRCMOD files

These codes are written in Python 2.7 and use T. Ben Thompson’s Okada wrapper, available at https://github.com/tbenthompson/okada_wrapper.

For a given SRCMOD file of interest, the code in ProcessSrcmodFile.py calculates stress changes, assembles aftershock locations into grid cells, and outputs the results to a CSV file. The function takes the name of the SRCMOD file as an argument (in FSP format; these files are all available for download at http://equake-rc.info/SRCMOD/searchmodels/allevents/).

For example, to process the SRCMOD file ‘s1992LANDER01COHE.fsp’, run:

> python ProcessSrcmodFile.py s1992LANDER01COHE.fsp

This will output a CSV file called 1992LANDER01COHE_grid.csv to the current working directory. This CSV file contains the stress changes at the centroids of grid cells (the size of which are specified ProcessSrcmodFile.py) as well as the aftershock counts within the cells. Other aftershock quantities are also assembled (magnitudes, etc.) for use in ongoing projects.

The pickle file isc_rev.pkl can be downloaded at: https://drive.google.com/file/d/1Co1IlK7ejBIvb5mxi7H7qSyGB13JoXT2/view?usp=sharing

A portion of this code (lines 13-43, 88-418 of ReadSrcmod.py) is based on an early repository located at https://github.com/google/stress_transfer/tree/master/stress_transfer.
