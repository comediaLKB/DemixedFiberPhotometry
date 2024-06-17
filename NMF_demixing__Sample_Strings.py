# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:49:12 2023 (Caio) - This file must have the filename strings of the vide data
Adapted on Fri Jun 14 23:26:34 2024 (Caio) - Organized module to share (Github)

@author: Caio VAZ RIMOLI: https://github.com/RimoliCV 
         Complex Media Optics lab: https://github.com/comediaLKB
"""
"""
DESCRIPTION:
        Display the strings of the datasets to be analyzed with NMF_demixing__ProofOfPrinciple_V22_2020614.py
        
    TO DO:
        Uncomment or comment the data path and filename you will use and assign:
            root_path   = ''
            data_folder = ''
            data_name   = ''
"""
#%% NOTE (!) ###################################################################### %%#
# Pay attention on the slash '/' or '\\' orientation... it depends if it is Linux, Windows, Mac etc

# sys.path.append(root_path + '\\' + data_folderpath_01)          # Pay attention on the slash '/' or '\\' orientation
# sys.path.append('../' + data_folderpath_01)                     # Pay attention on the slash '/' or '\\' orientation
# sys.path.append('../NMF_lightfield/localization-shackhartmann') # Pay attention on the slash '/' or '\\' orientation

#####################################################################################
#%% SAMPLE STRINGS (Data filenames) TO BE USED IN DATA ANALYSIS

# Here use your data path. example (Windows):
data_root_path = 'C:\\_RIMOLI_\\'     ### folder root path (string)
data_folder = '__data__\\20221109'   ### folder name (string) of the folder that contains the data (.mat files)
data_name =  'data_09112022_001' ### filename (string) of .mat file



# by using this separeted code you may automatize the analysis of multiple datasets in different folders.
