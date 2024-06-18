# Demixing fluorescence time traces transmitted by multimode fibers (Python)
NMF analysis for Demixed Fiber Photometry
-  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12125030.svg)](https://doi.org/10.5281/zenodo.12125030)

Python codes for the Non-negative Matrix Factorization (NMF) results shown in: **C. V. Rimoli, C. Moretti, F. Soldevila, E. Bremont, C. Ventalon, and S. Gigan - 2024 - Demixing fluorescence time traces transmitted by multimode fibers** arXiv:2306.00695 [physics.optics] https://doi.org/10.48550/arXiv.2306.00695

------------------------------

Python codes (modules):

**(1) NMF_demixing__ProofOfPrinciple_v22_20240614.py** *Description:* Proof-of-principle code used in the manuscript
- Input files: *(i) raw_ensemble_video_data.mat ; (ii) ground_truth_patterns_traces_gt.mat ; (iii) NMF_demixing__Sample_Strings.py ; (iv) tools_NMF.py .*

**(2) NMF_demixing__Sample_Strings.py** *Description:* Folder and Filename texts (strings) of (i) the raw video data and (ii) the ground truth time traces in .mat (Matlab format).

**(3) NMF_example.py** *Description:* Toy code (a simple example) of NMF analysis written by Fernando SOLDEVILA used for fluorescence speckle demixing (fully evolved speckle/scattering pattern) 
- Input files: *(i) raw_ensemble_video_data.mat ; (ii) ground_truth_patterns_traces_gt.mat ; (iii) tools_NMF.py .*

(4) **tools_NMF.py** *Description:* Some plotting and image display tools written by Fernando SOLDEVILA that is used in (1) and (3).

Check also:
-  @RimoliCV https://github.com/rimoliCV ;
-  @cdbaself https://github.com/cbasedlf/optsim .


