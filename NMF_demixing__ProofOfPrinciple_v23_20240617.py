# -*- coding: utf-8 -*-
ModulePy = str('NMF_demixing__ProofOfPrinciple_v23_20240617.py') 

'''
Description: NMF analysis demixing spatio-temporal ensemble signals from Fiber Photometry ensemble video data (proof-of-principle analysis).
- Input files: 
(i) raw_ensemble_video_data.mat                      [videodata_ddmmyyyy_001.mat]
(ii) ground_truth_patterns_traces_gt.mat             [videodata_ddmmyyyy_001_gt.mat]
(iii) NMF_demixing__Sample_Strings.py [here you must open this file and edit the text by adding the location of the input .mat files]
(iv) tools_NMF.py 

-----------------------------------------------------------
@authors:  Caio VAZ RIMOLI: - https://github.com/RimoliCV  
           Complex Media Optics lab: https://github.com/comediaLKB
           (This code is an adapted version from Fernandos code "NMF_example.py" - see log versions in the end of the code) 
            -- check also: Fernando SOLDEVILA: https://github.com/cbasedlf , https://github.com/cbasedlf/optsim
'''
 ### CODE STEPS: ###
''' -------------------------------------------------------
0) Mise en Place: Importing what is needed to work - choose options
1) Load dataset (from Matlab)
2) Clean it a little bit (do binning, filter background, etc) - optional
3) Perform NMF on the video
4) Display 1st results using 1st descending order 
    (that can assign/classify repeated NMF traces to different GT)
5) Get (GT,NMF)fingerprints & (GT,NMF)traces based on the highest correlation of the traces
     without repeating (GT,NMF)components - Special Descending Order (SD-order)
6) Display the results in SD-order (Special Descending SD order: most correlated NMF-GT results without repeating NMF results)
7) Display all the correlation plots (First descending FD, Special descending SD)orders of 'GT-GT', 'NMF-NMF', and 'GT-NMF' 
8) Display the Diagnose plot (to compare all the sorted results with NMF hierarchical outcome correlated with Ground Truth)
S) Save the results
V) Version log
-------------------------------------------------------------
'''


#%% ###################################################################### %%#
#%% [0.00]: Mise en Place: Import libraries, include datapath info, etc.
# import sys
import os
# from os import path
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
from sympy import simplify as checkif
# import scipy.signal
# from skimage import restoration
from PIL import Image
from sklearn.decomposition import NMF
import tools_NMF as tNMF
#import sample_strings3 as SS
import NMF_demixing__Sample_Strings as SS
import datetime
[start, start2] = [datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
print('Analys: A'+ start + ' began at: ' + start2)
# ''' ----------------------------------------------------------------------------------------------------- '''
# [0.01]: Choose what the code will do (switch True/False the switches): 
do_saving = False                    # if True, go to [0.02]: Choose how to switch the switches, and customize it
do_cropping = False                 # if True, go to [0.02]: Choose how to switch the switches, and customize it
do_binning = True                   # if True, go to [0.02]: Choose how to switch the switches, and customize it
do_envelop_filtering = False        # if True, go to [0.02]: Choose how to switch the switches, and customize it
make_rank_sameas_numbeads = True    # if False, go to [0.02]: Choose how to switch the switches, and customize it
add_BGD_to_rank = True              # if True, it makes: rank = rank +1.
remove_initial_frames = False       # if True, go to [0.02]: Choose how to switch the switches, and customize it
remove_single_frame = False          # if True, go to [0.02]: Choose how to switch the switches, and customize it
# remove_range_of_frames = False     # if True, go to [0.02]: Choose how to switch the switches, and customize it
norm_factor = False
bad_trace_in_red = False

#%%
# ##################################################################################################### %%#
# [0.02]: Choose how to switch the switches (Parameter settings):
''' ----------------------------------------------------------------------------------------------------- '''
if do_binning is True:          # Here, BINNING = 3 makes the analysis faster with the same quality as BINNING = 1
    BINNING = 3 #bin size
else:
    BINNING = 1 # without binning pixels
    
if do_cropping is True:     # Here we choose if we want to crop the video
    x_pxls = 600
    y_pxls = 600
   

if do_saving is True:       # Here we choose where to save the NMF proof of principle analysis
    Analysislog_root_path = 'C:\\__RIMOLI\\- LKB PostDoc\\_DATA\\'
    Analysislog_folder = '2024.06.06 - Analysis - 6 bead - Figures for NatComm\\'

 
if make_rank_sameas_numbeads is False:       # Here we chose if thoe NMF rank is exactly the number of the sources (numbeads)
    other_rank = 7     # if "add_BGD_to_rank = True" , then rank = other_rank + 1  
    
if remove_initial_frames is True:            # Here we choose if we want to remove some video frames from the NMF analysis
    first_frames = 50
else:
    first_frames = 0

if remove_single_frame is True:             # Here we choose if we want to remove a single frame from the video
    frame_to_delete = 618

if norm_factor is True:     # Here if we want to normalize the NMF traces to visually compare with GT traces
    norm_factor=0.632       # this was a test to normalize the time traces with a different value (visual display)
else:
    norm_factor=1
    



#%% ##################################################################################################### %%#
#%% [1.00]: SELECTING DATA PATH (IMPORTING THE STRINGS WHERE THE DATA ARE) ------------------------------- '''
'''--------------------------------------------------------------------------------------------------------'''
# Getting the strings of the Root_folder, the main experiment folder, and the filename (without .mat extension) of the data
data_root_path = SS.data_root_path
data_folder = SS.data_folder
data_name = SS.data_name

# Concatenating the full filename of the NMF and GT data (full filename str including root path)
GT_data_str = str(data_root_path + data_folder + '\\' + data_name + '_gt.mat')
NMF_data_str = str(data_root_path + data_folder + '\\' + data_name + '.mat')

# ##################################################################################################### %%#
# [1.01]: Load a dataset & its ground truth activations
''' ----------------------------------------------------------------------------------------------------- '''
print('Loading the data...')
#load ground truth (GT)
fgt = sio.loadmat(GT_data_str)
# fgt = sio.loadmat('data_09112022_001_gt.mat') # Another way of loading files
act_gt = fgt['pat'].T 
numbeads = act_gt.shape[1] #get real number of beads (from ground truth)

# OPTIONAL: removing one single frame from the dataset -- the GT part
if remove_single_frame is True:
    old_act_gt = act_gt;
    print('You chose to delete a frame:', frame_to_delete)
    act_gt = np.delete(act_gt, frame_to_delete, axis=0);
    act_gt_norm = np.zeros(act_gt.shape)
    #first_frames=1
    #matrix= act_gt[first_frames::]
    matrix=[];
    matrix= act_gt;
    for idx in range(0,act_gt.shape[1]):
        act_gt_norm[:,idx] = (matrix[:,idx]-np.min(matrix[:,idx]))/(norm_factor*np.max(matrix[:,idx])-np.min(matrix[:,idx]+1e-12))
        pass
    
    
# Normalize ground truth traces (for doing comparisons later) - needs to have the same length to compare
act_gt_norm = np.zeros(act_gt[first_frames::].shape)
if remove_initial_frames is False:
    # act_gt_norm = tNMF.norm_dimension(act_gt, dim = 1, mode='0to1') #normalization by the maximum
    matrix= act_gt[first_frames::]
    for idx in range(0,act_gt.shape[1]):
        act_gt_norm[:,idx] = (matrix[:,idx]-np.min(matrix[:,idx]))/(norm_factor*np.max(matrix[:,idx])-np.min(matrix[:,idx]+1e-12))
        pass
else:
    # act_gt_norm = tNMF.norm_dimension(act_gt[first_frames::], dim = 1, mode='0to1') #normalization by the maximum
    matrix= act_gt[first_frames::]
    for idx in range(0,act_gt.shape[1]):
        act_gt_norm[:,idx] = (matrix[:,idx]-np.min(matrix[:,idx]))/(norm_factor*np.max(matrix[:,idx])-np.min(matrix[:,idx]+1e-12))
        pass
    

# Load NMF dataset
f = h5py.File(NMF_data_str, mode = 'r')
# f = h5py.File('data_09112022_001.mat', mode = 'r')
video = np.array(f['video_data'])
video = np.moveaxis(video,0,2) #rearrange to px*px*time form
print('Done!')
print(' ')

# OPTIONAL: removing one single frame from the dataset -- the raw data part
if remove_single_frame is True:
    old_video = video;
    time_index_to_remove = frame_to_delete + numbeads +1;
    print('Adjusting the data video!')
    video = np.delete(video, time_index_to_remove, axis=2)

# ##################################################################################################### %%#
# [1.02]: Choose how to minimally cropp the raw video (rounding the FOV pixel number):
''' ----------------------------------------------------------------------------------------------------- '''
if do_cropping is False:
    x_pxls = int(mt.floor(video.shape[0]/10))*10      # rounding down the Nb pixels: if 697 it will be rounded to 690
    y_pxls = int(mt.floor(video.shape[1]/10))*10 
    
       
#%% ##################################################################################################### %%#
#%% [2.00]: Clean dataset: binning (to go faster) + envelop filtering (optional)
''' -------------------------------------------------------------------------------------------------------'''
print('Preprocessing the dataset...')

# [2.01]: Crop (or not) the central part of the images
typical_size = [x_pxls,y_pxls] 
frame_size = np.asarray((typical_size), dtype = int)

# Check orientation of the frame (vertical video or horizontal)
if frame_size[0] > frame_size[1]:
    ORIENTATION = 'vertical'
elif frame_size[0] < frame_size[1]:
    ORIENTATION = 'horizontal'
else:
    ORIENTATION = 'square'

# Preallocate cropped video
video_small = np.zeros((frame_size[0], frame_size[1], video.shape[2]),
                       dtype = 'uint16')
# Crop the video
for idx in range(video.shape[2]):
    video_small[:,:,idx] = tNMF.cropROI(video[:,:,idx], size = frame_size,
                                        center_pos = (int(frame_size[0]/2),
                                                    int(frame_size[1]/2)))

# [2.02] Do the binning
# calculate new frame sizes after binning
new_frame_size = (frame_size/BINNING).astype(int)
short_side = np.min(new_frame_size) #calculate short side
long_side = np.max(new_frame_size) #calculate long side

# preallocate binned video
video_binned = np.zeros((new_frame_size[0], new_frame_size[1], video.shape[2]))
#do the binning
print('Binning...')
print(' ')
for idx in range(video.shape[2]):
    temp = video_small[:,:,idx]
    #temp = Image.fromarray(temp).resize((new_frame_size[1], new_frame_size[0]),
    #                                    resample = Image.NEAREST) ### Image.NEAREST is DEPRECATED
    temp = Image.fromarray(temp).resize((new_frame_size[1], new_frame_size[0]),
                                        resample = Image.Resampling.NEAREST)
    
    temp = np.array(temp)
    video_binned[:,:,idx] = temp
print('Done!')
print(' ')


# [2.03]: Filter low frequency background (envelope)  ### This was used on fully eveolved speckles, it was not used for MMF patterns
video_highpass = np.zeros(video_binned.shape) # preallocate
print('Checking if we will filter the envelope... ')     ### Multimode fiber patterns do not need to remove gaussian envelope
if do_envelop_filtering == 1:
    print('Yep...')
    #Build gaussian filter. Sizes have to be manually tuned for now
    gaussian = tNMF.buildGauss(short_side, (3,3),
                                (int(short_side/2), int(short_side/2)), 0)
    #pad the filter so it gets same size as the frame
    #(gaussian is always a square image, but video might be rectangular)
    if ORIENTATION == 'vertical':
        padsize = long_side - short_side
        #take care of padsize, otherwise filter might be smaller than the image
        if tNMF.iseven(padsize):
            gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2)), (0,0)))
        else:
            gaussian = np.pad(gaussian, ((int(padsize/2), int(padsize/2) + 1), (0,0)))
    elif ORIENTATION == 'horizontal':
        padsize = long_side - short_side
        #take care of padsize, otherwise filter might be smaller than the image
        if tNMF.iseven(padsize):
            gaussian = np.pad(gaussian, ((0,0), (int(padsize/2), int(padsize/2))))
        else:
            gaussian = np.pad(gaussian, ((0,0), (int(padsize/2), int(padsize/2) + 1)))
    else: #if is neither hor. nor vert., is square and no need to do anything
        pass
    lowfreqfilter = 1 - gaussian
    #do the filtering
    for idx in range(0,video.shape[2]):
        video_highpass[:,:,idx] = tNMF.filt_fourier(video_binned[:,:,idx],
                                                    lowfreqfilter)
    print('Envelope filtering finished. ')

else:
    print('No...')
    gaussian = []
    video_highpass = video_binned
    print('Envelope filtering was not applied. ') # typical case for multimode fiber fiber photometry demixing

#%% ##################################################################################################### %%#
#%% [3.00] Do NMF on the pre-processed video (croped,binned, and filtered)
print('Starting the NMF...')
#numbeads = act_gt.shape[1] #get real number of beads (from ground truth)
initial_frames = numbeads + first_frames

# Here the analysis will be different depending on the expected rank:
if make_rank_sameas_numbeads is True:
    if add_BGD_to_rank is True:
        rank = numbeads + 1      #set rank as number of beads + background
    else:
        rank = numbeads          #set rank = exact number of sources
else:
    if add_BGD_to_rank is True:
        rank = other_rank + 1      #set rank as the chosen rank + background
    else:
        rank = other_rank          #set rank as the chosen rank

# remove ground truth frames (initial part of the acquired video in the lab)
X = video_highpass[:,:,initial_frames::].copy()

# reshape intro matrix form
X = np.reshape(X, (new_frame_size[0] * new_frame_size[1], X.shape[2]))

# Do the NMF without a priori knowledge in the initialization init='nndsvd'
 # Create the model:
  # Parameters of the model (that will be saved in the h5py file)  
model_n_components = rank 
model_init = 'nndsvd'
model_random_state = 0
model_max_iter = 3000
model_solver = 'cd'
model_l1_ratio = 1   # 0.5
model_beta_loss = 2  
model_verbose = 0 
model_alpha_W = 0    # 1.5
model_alpha_H = 0    # 0.5

model = NMF(n_components        = model_n_components, 
            init                = model_init,
            random_state        = model_random_state,
            max_iter            = model_max_iter, 
            solver              = model_solver,
            l1_ratio            = model_l1_ratio,
            beta_loss           = model_beta_loss,
            verbose             = model_verbose,
            alpha_W             = model_alpha_W,
            alpha_H             = model_alpha_H)

#Run the model, store spatial fingerprints in W:
W = model.fit_transform(X)
#Store temporal activities in H:
H = model.components_
#Reshape spatial fingerprints in 3D tensor form:
fingerprints = np.reshape(W, (new_frame_size[0], new_frame_size[1], rank))
#Store ground truth fingerprints, for comparison later
fingerprints_gt = video_highpass[:,:,0:numbeads]

tNMF.show_Nimg(fingerprints_gt) #show ground truth fingerprints
tNMF.show_Nimg(fingerprints) #show recovered fingerprints
print('NMF finished...')
print(' ')
#%% ############################################################################################################################## %%#
#%% [4.00]: Plot of the results after 1st ordering: NMF that most correlated with GT traces (Repetition of NMFs traces are possible at this point! Need to avoid repetition later with SD-order)
# ----------  Identifying the recovered traces and plotting them with the ground truth -------------------------------------------- #
print('Plotting the temporal traces...')
print(' ')
act = H.T
# act_norm = tNMF.norm_dimension(act, dim = 1, mode = '0to1') #normalize recovery

act_norm = np.zeros(act.shape)
# norm_factor here is just to visually emphasize/highight the traces' peak in the normalized temporal activity. Default: norm_factor = 1
for idx in range(0,act.shape[1]):
    act_norm[:,idx] = (act[:,idx]-np.min(act[:,idx]))/(norm_factor*np.max(act[:,idx])-np.min(act[:,idx]+1e-12))
    pass



''' # 1st Ordering/Sorting step: Explained Here! #

1) Take each recovered trace
2) Correlate with the ground truth
3) identify correspondence by looking at the maximum of the correlation.
4) Do that with all the traces, so in the end you get as many pairs of 
    recovery-ground truth as the rank of the system.
    
(!) Important note (!)
Sometimes this gives errors (if the traces from the NMF are not very good,
 you might "classify" 2 traces from the NMF to the same ground truth trace) - repetition - it will be corrected later in the code
'''

ordering_GTidx_fullsize = np.zeros(numbeads, dtype = 'int')        #preallocating
ordering_NMFidx_fullsize = np.zeros(rank, dtype = 'int')          #preallocating
r1 = np.zeros((numbeads, rank)) #GT-NMF trace correlation coeff (preallocating)

for idx in range(numbeads):
    #pick ground truth temporal trace
    temp_gtruth = act_gt_norm[:,idx]
    #pick ground truth index
    ordering_GTidx_fullsize[idx] = idx
    #Find equivalent eigenvector in NMF reconstruction (normalized)
    for tidx in range(rank):
        #Compute correlation between NMF traces (tidx) and the current GT temporal trace (idx)
        # Pick element (1,0) of the correlation matrix (correlation between the two vectors)
        r1[idx,tidx] = np.corrcoef(temp_gtruth, act_norm[:,tidx])[1,0]             # GT-NMF corr coefs without sorting/ordering
    #Identify the NMF traces that match the current temporal trace by using
    #the maximum correlation value among traces #
    #If the chosen NMF rank is bigger than the number of GT sources...
    if rank >= numbeads :
        #Do the the 1st ordering -> it might repeat NMF components (!) 
        #ordering_NMFidx_fullsize[idx] = int(np.where(r1[idx,:] == np.max(r1[idx,:]))[0])
        #ordering_NMFidx_fullsize[idx] = np.where(r1[idx,:] == np.max(r1[idx,:]))[0].astype(int)        # .astype(int) here converts "[array[index], dtype]" format into "[index]"
        ordering_NMFidx_fullsize[idx] = np.argmax(r1[idx, :]).astype(int)
        
    else:
        # Do the same until the index of GT is equal to the index of NMF rank
        if idx <= tidx:
            #ordering_NMFidx_fullsize[idx] = np.where(r1[idx,:] == np.max(r1[idx,:]))[0].astype(int)       # .astype(int) here converts "[array[index], dtype]" format into "[index]    
            ordering_NMFidx_fullsize[idx] = np.argmax(r1[idx, :]).astype(int)
            
    
    
if rank >= numbeads:
    # When: Number of the NMF rank is bigger than the number of GT sources
    # I must truncate the  ordering_NMFidx_fullsize so that both (GT numbeads, NMF rank) have the same number of elements
    # I must truncate the ordering vector so that both (GT numbeads, NMF rank) have the same number of elements
    ordering_GTidx = ordering_GTidx_fullsize
    ordering_NMFidx = np.split(ordering_NMFidx_fullsize, [numbeads,-1])[0]
    print('1st Order GT sequence:')
    print(str(ordering_GTidx_fullsize))
    print('1st Order NMF sequence (all ranks):')
    print(str(ordering_NMFidx_fullsize))
    print('1st Order NMF sequence (rank = numbeads sources):')
    print(ordering_NMFidx)     
else: 
    # When: Number of the NMF rank is smaller than the number of GT sources
    # I must truncate the  ordering_GTidx so that both (GT numbeads, NMF rank) have the same number of elements
    # The ordering vector forces to truncate the GT sources in order to (GT numbeads, NMF rank) have the same number of elements
    ordering_GTidx = np.split(ordering_GTidx_fullsize, [rank,-1])[0]
    ordering_NMFidx = ordering_NMFidx_fullsize
    print('1st Order GT sequence (all sources):')
    print(str(ordering_GTidx_fullsize))
    print('1st Order GT sequence (rank = numbeads sources):')
    print(str(ordering_GTidx))
    print('1st Order NMF sequence:')
    print(ordering_NMFidx) 

                      
ordering_1st = np.vstack((ordering_GTidx, ordering_NMFidx)).T # this only works when rank == numbeads (after truncation)
print('1st Order array:')
print(str(ordering_1st))

#%% 
# ##################################################################################################### %%#
# [4.01]: Plotting the NMF fingerprints (1st ordering) - these plots can have repetition 
''' --------------------------------------------------------------------------------- '''
#re-order spatial fingerprints to match ground truth order, then show
fingerprints_ordered = fingerprints[:,:,ordering_NMFidx].copy()
#show recovered fingerprints
tNMF.show_Nimg(fingerprints_ordered[:,:,0:numbeads])
#show recovered fingerprints
# tNMF.show_Nimg(fingerprints_ordered)

#%%
# ##################################################################################################### %%#
# [4.02]: Plotting the temporal activity traces (1st ordering) - these plots can have repetition 
''' --------------------------------------------------------------------------------- '''

#Swap order to match ground truth ordering
act_ordered = act_norm[:,ordering_NMFidx]
#create colormap (to ease visualization)
colormap = plt.cm.nipy_spectral
colors = [colormap(idx) for idx in np.linspace(0, 1, 4 * numbeads)]
#Plot all traces
fig,ax = plt.subplots(figsize = (8,10))
ax.set_prop_cycle('color', colors)
SPACING = 2 #spacing between pairs of traces

if rank >= numbeads:
    truncated_range = numbeads      # The numbeads limits the number of unique fingerprints
    bigger_range = rank
else:
    truncated_range = rank      ## The NMF rank limits the number of unique fingerprints
    bigger_range = numbeads
    
for idx in range(truncated_range):
    ax.plot(act_gt_norm[:,idx] + idx * SPACING, label = 'G.T. trace #' + str(idx + 1))
    # ax.plot(act_gt[:,idx] + idx * SPACING, label = 'G.T. trace #' + str(idx + 1))
    ax.plot(act_ordered[:,idx] + idx * SPACING + 0.6,
            label = 'NMF trace #' + str(idx + 1))
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Intensity (a.u.)')
plt.suptitle('Temporal activity recovery via NMF - Recovered traces might repeat')
# plt.legend()
plt.tight_layout()
plt.show()


## OPTIONAL: Plot all recovered NMF traces (without the ground truth ones for comparison)
# fig,ax = plt.subplots(figsize = (8,10))
# #create colormap (to ease visualization)
# colormap = plt.cm.nipy_spectral
# colors = [colormap(idx) for idx in np.linspace(0, 1, rank)]
# ax.set_prop_cycle('color', colors)
# SPACING = 2 #spacing between pairs of traces
# for idx in range(rank):
#     # ax.plot(act_gt[:,idx] + idx * SPACING, label = 'G.T. trace #'+ str(idx + 1))
#     ax.plot(act_norm[:,idx] + idx * SPACING, label = 'NMF trace #' + str(idx + 1))
# ax.set_xlabel('Time (a.u.)')
# ax.set_ylabel('Intensity (a.u.)')
# plt.suptitle('Temporal activity recovery via NMF)
# # plt.legend()
# plt.tight_layout()
# plt.show()
print('Temporal traces plotted...')

#%% ##################################################################################################### %%#
#%% [5.00]: Re-ordering the results withou any repetition (Special Descending order, SD-order)
''' --------------------------------------------------------------------------------- '''

''' # SD Ordering/Sorting step: Explained Here! #
# SD == Swap Descending, Special Descending order:  
# Now we avoid the repeated classifications of 2 NMFs to the same GT
 
1) Copy the r1 (correlation matrix coeficients in a 2Darray) into a DataFrame (corr_df)
    (to better operate on rows and columns of the correlation matrix)
5) Create a temporary DataFrame (temp_df) to be updated in a loop, temp_df = corr_df.copy())
2) Find the (1D) argmax_index (i) of the most correlated GT-NMF pair in temp_df
3) Find the correspondent 2D_indexes (gt_i,nmf_i) of that (1D) argmax_index (i)
4) Append (save) the 2D_indexes in a new array
5) Replace with NaNs all the elements in the row 'gt_i' and in the 'nmf_i'
6) Assign this new DataFrame with NaNs in 'temp_df'
7) Repeat steps 2'-5' on 'temp_df' until you finish the number of the GT sources (numbeads)
8) In the end, you accumulate the descending order of the array [gt_i, nmf_i] of the correlation table
    without repeating the same index (due to NaN values) in the subsequent run
9) Finally you swapp all the GT-NMF components following the arrays [gt_i, nmf_i] order to get 
    the Hierarchical display of fingerprints, traces, and correlation coeficient plots
'''

# SD-Ordering (Hierarchical order without repetition)

idx_labels = [None]*numbeads    # preallocating list 
col_labels = [None]*rank        # preallocating list 

for gt_idx in np.arange(numbeads):
    for nmf_idx in np.arange(rank):
        idx_labels[gt_idx] = str('GT_' + str(gt_idx))
        col_labels[nmf_idx] = str('NMF_' + str(nmf_idx))
     
corr_df = pd.DataFrame(r1, index=idx_labels, columns=col_labels) 
rmax_t = np.zeros(numbeads)     # numbeads --> truncated_range 
cmax_t = np.zeros(rank)         # rank --> bigger_range 

df_t_before = corr_df.copy()

for df_idx in range(truncated_range):           # numbeads --> truncated_range
    # print('df_idx', df_idx)
    df_t = df_t_before.copy()
    # print(df_t_before)
    row_idx = np.argmax(np.max(df_t, axis=1))          ### get the 1D index of the maximum value of a df or array row
    col_idx = np.argmax(np.max(df_t, axis=0))          ### get the 1D index of the maximum value of a df or array column
    [rmax_t[df_idx],cmax_t[df_idx]]=[row_idx,col_idx]  # storing the indexes of the maximum value of the array 
    df_t_before.iloc[row_idx,:]=float('nan')             ### blocking the previous row    of the max with Nan
    df_t_before.iloc[:,col_idx]=float('nan')             ### blocking the previous column of the max with Nan
    # Nan_row = df_t_before.iloc[row_idx,:]=float('nan')  ### blocking the previous row    of the max with Nan
    # Nan_col = df_t_before.iloc[:,col_idx]=float('nan')  ### blocking the previous column of the max with Nan
    # print(row_idx,col_idx)
    # print(df_t)
    
SD_order_GT_fullsize = rmax_t.astype(int)              # converting float values of the sequence the to int value
SD_order_NMF_fullsize = cmax_t.astype(int)        # converting float values of the sequence the to int values

# I must truncate the  SD_order_GT_fullsize and SD_order_NMF_fullsize so that both (GT numbeads, NMF rank) have the same number of elements
SD_order_NMF = np.split(SD_order_NMF_fullsize, [truncated_range,-1])[0]     # numbeads --> truncated_range
# I must truncate the  SD_order_GT_fullsize and SD_order_NMF_fullsize so that both (GT numbeads, NMF rank) have the same number of elements
SD_order_GT = np.split(SD_order_GT_fullsize, [truncated_range,-1])[0]       # numbeads --> truncated_range


# display what it was done at this step 
print('1st Order GT sequence (r1)')
print(ordering_GTidx)
print('1st Order NMF sequence (r1)')
print(ordering_NMFidx)
print(' ')
print('GT-idx in Hierarhical SD-order:')
print(SD_order_GT)
print('NMF-idx in Hierarchical SD-order:')
print(SD_order_NMF)


''' APPLYING THE SD-ORDER on the temporal traces and on the spatial fingerprints'''
# Re-ordering [SD-order] the TEMPORAL ACTIVITY: GT-traces and NMF-traces
act_gt_sd = act_gt_norm[:,SD_order_GT] # SD-order (Top down in the traces plot)
act_nmf_sd = act_norm[:,SD_order_NMF]  # SD-order (Top down in the traces plot)

# Re-ordering [SD-order] the SPATIAL FINGERPRINTS: GT-fingeprints and NMF-fingerprints 
fingerprints_gt_sd      = fingerprints_gt[:,:,SD_order_GT].copy()
fingerprints_nmf_sd     = fingerprints[:,:,SD_order_NMF].copy()


# ##################################################################################################### %%#
# [5.01]: Plotting the NMF fingerprints (SD ordering)
''' --------------------------------------------------------------------------------- '''
#show GT in Hierarchical SD-order
tNMF.show_Nimg(fingerprints_gt_sd[:,:,0:truncated_range])   # numbeads --> truncated_range
#show recovered NMF fingerprints in Hierarchical SD-order 
tNMF.show_Nimg(fingerprints_nmf_sd[:,:,0:truncated_range])  # numbeads --> truncated_range

#%% ##################################################################################################################### %%#
#%% [6.00] Plotting all the activity traces in Hierarchical SD-order (Top: the highest correlation. Bottom: the lowest correlated coefs)
'''---------------------------------------------------------------------------------------------------------------------- '''

# ##################################################################################################### %%#
# [6.01]: Inverting the SD-order for a nice display when Plotting the Temporal Activity Traces (I-order)
''' ---------------------------------------------------------------------------------------------------- '''

''' I had to create the inversed order to give a nice Figure plot of the traces'''
I_order_GT = SD_order_GT[::-1]      # inverting the order to make looks good in the plot
I_order_NMF = SD_order_NMF[::-1]    # inverting the order to make  looks good in the plot
print(' ')
print('GT-idx in Inverted I-order:')
print(I_order_GT)
print('NMF-idx in Inverted I-order:')
print(I_order_NMF)

# Inverting [I-order] the SD-order to better plot the TEMPORAL ACTIVITY
act_gt_inv = act_gt_norm[:,I_order_GT] # I stands for Inverted order to plot traces
act_nmf_inv = act_norm[:,I_order_NMF]  # I stands for Inverted order to plot traces

# Inverting [I-order] the SD-order to better plot the SPATIAL FINGERPRINTS
fingerprints_gt_inv     = fingerprints_gt[:,:,I_order_GT].copy()
fingerprints_nmf_inv     = fingerprints[:,:,I_order_NMF].copy()

#%%
# ##################################################################################################### %%#
# [6.02]: Display Temporal Activity Traces Hierarchical (I-order): Top traces: highest correlation
''' ---------------------------------------------------------------------------------------------------- '''

#create colormap (to ease visualization)
# colormap = plt.cm.nipy_spectral
colormap = plt.cm.winter
colors_NMF = [colormap(idx) for idx in np.linspace(0, 1, 2 * truncated_range)]  # numbeads --> truncated_range
colors_GT = np.zeros(np.shape(colors_NMF))
colors = np.append(colors_GT, colors_GT, axis=0)

rgb_NMF_idx = 0
rgb_GT_idx = 0

for idx in np.arange(4 * truncated_range):  # numbeads --> truncated_range (truncated_range works for any chosen rank)
    if checkif(idx).is_even:
        colors[idx,:] = list(colors_NMF[rgb_NMF_idx])
        rgb_NMF_idx= rgb_NMF_idx +1
        print('NMF even')
        print(idx)
        print(rgb_NMF_idx)
    else:
     if checkif(idx).is_odd:
        colors[idx,:] = [0.5,0.5,0.5,1.0]
        print('GT odd')
        print(idx)
     else:
        pass

#Plot all traces
fig_h = 8    # figure high size
fig_w = 10   # figure width size
SPACING = 1.3  # spacing between pairs of traces
Tidx_shift = 0  # Legend: start counting the fingerprints from 0 or 1?
peak_amp_GT = 0.5
peak_amp_NMF = 1

fig,ax = plt.subplots(figsize = (fig_h,fig_w))
ax.set_prop_cycle('color', colors)
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Intensity (a.u.)')
plt.suptitle('Sorted temporal activity recovery via NMF')

if bad_trace_in_red is False:
# Plotting the traces that went right in winter colormap :
    ''' If we plot in the inversed SD oder, the top traces are the ones with highest correlation coeficients'''
    for idx in range(0,truncated_range):     # numbeads --> truncated_range
        ax.plot(peak_amp_NMF*act_nmf_inv[:,idx] + 1 + idx * SPACING + 0.21, label = 'NMF trace #' + str(I_order_NMF[idx] + Tidx_shift))
        ax.plot(peak_amp_GT*act_gt_inv[:,idx] + 1 + idx * SPACING, label = 'G.T.  trace #' + str(I_order_GT[idx] + Tidx_shift))
else:
# Plotting the last trace that went wrong in red:
    red_trace = 0
    for idx in range(1,truncated_range):     # numbeads --> truncated_range
        ax.plot(peak_amp_NMF*act_nmf_inv[:,idx] + 1 + idx * SPACING + 0.21, label = 'NMF trace #' + str(I_order_NMF[idx] + Tidx_shift))
        ax.plot(peak_amp_GT*act_gt_inv[:,idx] + 1 + idx * SPACING, label = 'G.T.  trace #' + str(I_order_GT[idx] + Tidx_shift))
    
    ax.plot(peak_amp_NMF*act_nmf_inv[:,0] + 1 + 0 * 1 + 0.11, color='r' ,
            label = 'NMF trace #' + str(I_order_NMF[idx] + Tidx_shift))
    ax.plot(peak_amp_GT*act_gt_inv[:,0] + 1 + 0 * 1, color=[0.5,0.5,0.5,1.0] ,
            label = 'G.T.  trace #' + str(I_order_GT[idx] + Tidx_shift))


#%% Adjusting details of the figure

''' Adjusting the legend organization'''
# labelspacing= (mt.e/2)*(fig_w/fig_h)*(2*truncated_range)/fig_h  # numbeads --> truncated_range
# labelspacing = 1
# handles,labels = ax.get_legend_handles_labels()
# legend_order = np.arange(0,len(labels))[::-1]
# ax.legend([ handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order],\
#           bbox_to_anchor=(1.0, 1.005), labelspacing=labelspacing)

Nb_x = video[0,0,:].shape[0] 
Nb_y = 2*truncated_range + 1       # numbeads --> truncated_range
# Nb_y = truncated_range + 1       # numbeads --> truncated_range
x_ticks_range = (mt.floor(Nb_x/100))*10
xticks = np.arange(0, Nb_x, x_ticks_range)
yticks = np.arange(0, Nb_y, 1)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
plt.yticks(color='w')
plt.tight_layout()
plt.show()


#%% ############################################################################# %%#
#%% [7.00]: CREATING THE CORRELATION PLOTS (GT-GT, NMF-NMF, GT-NMF) in SD-order 
'''------------------------------------------------------------------------------ '''

temp_gtruth=[]                                  #preallocating
temp_NMF=[]                                     #preallocating
corr_GT_GT_sd = np.zeros((truncated_range,truncated_range))   #preallocating  # numbeads --> truncated_range
corr_NMF_NMF_sd = np.zeros((truncated_range,truncated_range)) #preallocating  # numbeads --> truncated_range
corr_GT_NMF_sd = np.zeros((truncated_range,truncated_range))  #preallocating  # numbeads --> truncated_range
corr_NMF_GT_sd = np.zeros((truncated_range,truncated_range))  #preallocating  # numbeads --> truncated_range

# Finding all correlations:
for Ridx in range(truncated_range):            # numbeads --> truncated_range
    temp_gtruth = act_gt_sd[:,Ridx]
    temp_NMF = act_nmf_sd[:,Ridx]
    for Cidx in range(truncated_range):          # numbeads --> truncated_range
        # Putting GT in rows (along Y axis) and NMF in columns (along X axis)
        corr_GT_GT_sd[Ridx,Cidx]   = np.corrcoef(temp_gtruth , act_gt_sd[:,Cidx])[1,0]
        corr_NMF_NMF_sd[Ridx,Cidx] = np.corrcoef(temp_NMF    , act_nmf_sd[:,Cidx])[1,0]
        corr_GT_NMF_sd[Ridx,Cidx]  = np.corrcoef(temp_gtruth , act_nmf_sd[:,Cidx])[1,0]
        corr_NMF_GT_sd[Ridx,Cidx]  = np.corrcoef(temp_NMF    , act_gt_sd[:,Cidx])[1,0]


# Making the Figures:
# # ---------------------------------------------------------- #
fig0, ax = plt.subplots(figsize = (8,5))
im0 = ax.imshow(r1, cmap = 'seismic', vmin = -1, vmax = 1)
fig0.colorbar(im0)
plt.title('r1 - GT-NMF-norm correlations')
plt.show()
print('r1 - GT-NMF-norm')
# print(r1)
print(np.round(r1,2))
# # ---------------------------------------------------------- #

# # ---------------------------------------------------------- #
# fig1, ax = plt.subplots(figsize = (8,5))
# im1 = ax.imshow(r1.T, cmap = 'seismic', vmin = -1, vmax = 1)
# fig1.colorbar(im1)
# plt.title('r1.T - GT-NMF-norm correlationss')
# plt.show()
# print('r1.T - GT-NMF-norm')
# # print(r1.T)
# print(np.round(r1.T,2))
# # ---------------------------------------------------------- #

# # ---------------------------------------------------------- #
# fig2, ax = plt.subplots(figsize = (8,5))
# im2 = ax.imshow(corr_df, cmap = 'seismic', vmin = -1, vmax = 1)
# fig2.colorbar(im2)
# plt.title('corr_df correlations')
# plt.show()
# print('corr_df')
# # print(corr_df)
# print(np.round(corr_df,2))
# # ---------------------------------------------------------- #

# # ---------------------------------------------------------- #
fig6, ax6 = plt.subplots(figsize = (8,5))
im6 = ax6.imshow(corr_GT_GT_sd, cmap = 'seismic', vmin = -1, vmax = 1)
fig6.colorbar(im6)
plt.title('SD(no-rep): corr_GT_GT_sd')
plt.show()
print('SD corr_GT_GT_sd')
# print(corr_GT_GT_sd)
print(np.round(corr_GT_GT_sd, 2))
# # ---------------------------------------------------------- #

# # ---------------------------------------------------------- #
fig7, ax7 = plt.subplots(figsize = (8,5))
im7 = ax7.imshow(corr_NMF_NMF_sd, cmap = 'seismic', vmin = -1, vmax = 1)
fig7.colorbar(im7)
plt.title('SD(no-rep): corr_NMF_NMF_sd')
plt.show()
print('SD corr')
# print(corr_NMF_NMF_sd)
print(np.round(corr_NMF_NMF_sd, 2))
# # ---------------------------------------------------------- #

# # ---------------------------------------------------------- #
fig8, ax8 = plt.subplots(figsize = (8,5))
im8 = ax8.imshow(corr_GT_NMF_sd, cmap = 'seismic', vmin = -1, vmax = 1)
fig8.colorbar(im8)
plt.title('SD(no-rep): corr_GT_NMF_sd')
plt.show()
print('SD corr_GT_NMF_sd')
# print(corr_GT_NMF_sd)
print(np.round(corr_GT_NMF_sd, 2))
# # ---------------------------------------------------------- #

# # ---------------------------------------------------------- #
# fig9, ax9 = plt.subplots(figsize = (8,5))
# im9 = ax9.imshow(corr_NMF_GT_sd, cmap = 'seismic', vmin = -1, vmax = 1)
# fig9.colorbar(im9)
# plt.title('SD(no-rep): corr_NMF_GT_sd')
# plt.show()
# print('SD corr_NMF_GT_sd')
# # print(corr_NMF_GT_sd)
# print(np.round(corr_NMF_GT_sd, 2))
# # ---------------------------------------------------------- #

# Saving the diagonal of 'corr_GT_NMF_sd' into a array
SD_diagonal = np.zeros(np.shape(corr_GT_NMF_sd[0]))
for row_idx in range(truncated_range):                  # numbeads --> truncated_range
    SD_diagonal[row_idx] = corr_GT_NMF_sd[row_idx,row_idx]
SD_diagonal = SD_diagonal.reshape(-1,1)

#%% ############################################################################################## %%#
#%% [8.00]: DIAGNOSE PLOT (COMPLETE) - FINGERPRINT-TRACE pair with different orders ##################
# ---- Showing images of each temporal trace with its fingerprint ---------------------------------- #
# ---- (GT-trace, NMF-trace)pair, with orders: (1) Not ordered, (2) 1st Ordered, (3) SD Ordered ---- #
'''-----------------------------------------------------------------------------------'''
# --------------------- #
# ### output data ##### #
# --------------------- # 
# fingerprints_gt       #
# fingerprints          # 
# fingerprints_ordered  #
# fingerprints_gt_sd    # 
# fingerprints_nmf_sd   # 
# act_gt_norm           #
# act_norm              #
# act_ordered           #
# act_gt_sd             #
# act_nmf_sd            #
# ##################### #

nrows = truncated_range        # numbeads --> truncated_range
ncols = 6
all_cols = ncols + 4
fingerprint_width = 1
traces_width = 2
# fig, axs = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(40, 24), gridspec_kw={'width_ratios': [fingerprint_width,traces_width,fingerprint_width,traces_width,\
#                                                                                                      fingerprint_width,traces_width,fingerprint_width,traces_width,\   
fig, axs = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(40, 24), gridspec_kw={'width_ratios': [fingerprint_width,traces_width]*int((all_cols/2))})
for row_idx in range(nrows):
    ### Creating the subplot of each fingerprint-trace pair
    axs[row_idx][0].imshow(fingerprints_gt[:,:,row_idx])          # GT  fingerprint       (Raw/Filtered)
    axs[row_idx][1].plot(act_gt_norm[:,row_idx])                  # GT  activity traces   (Normalized)
    axs[row_idx][2].imshow(fingerprints[:,:,row_idx])             # NMF fingerprint       (Not ordered, Raw/filtered)
    axs[row_idx][3].plot(act_norm[:,row_idx])                     # NMF activity traces   (Not ordered, normalized)
    axs[row_idx][4].imshow(fingerprints_ordered[:,:,row_idx])     # NMF fingerprints      (1st order)
    axs[row_idx][5].plot(act_ordered[:,row_idx])                  # NMF activity traces   (1st order)
    axs[row_idx][6].imshow(fingerprints_gt_sd[:,:,row_idx])       # GT  fingerprints      (SD-ordered = Special Descending hierarchical based on NMF correlations, no rep)
    axs[row_idx][7].plot(act_gt_sd[:,row_idx])                    # GT  activity traces   (SD-ordered = Special Descending hierarchical based on NMF correlations, no rep)
    axs[row_idx][8].imshow(fingerprints_nmf_sd[:,:,row_idx])      # NMF fingerprints      (SD-ordered) 
    axs[row_idx][9].plot(act_nmf_sd[:,row_idx])                   # NMF_norm_traces       (SD-ordered)

    '''---------------------------------------------------------------------------------------------------------------'''
    ### Writing a title for each subplot                            
    axs[row_idx][0].set(title='fingerprints GT')                   # GT  fingerprint       (Raw/Filtered)
    axs[row_idx][1].set(title='act_gt_norm')                       # GT  activity traces   (Normalized)
    axs[row_idx][2].set(title='fingerprints NMF (Raw)')            # NMF fingerprint       (Not ordered, Raw/filtered) 
    axs[row_idx][3].set(title='act_norm (Not Ordered)')            # NMF activity traces   (Not ordered, normalized) 
    axs[row_idx][4].set(title='fingerprints NMF (r1)')             # NMF fingerprint       (1st order)
    axs[row_idx][5].set(title='act_1st_ordered (w/ rep)')          # NMF activity traces   (1st order)
    axs[row_idx][6].set(title='fingerprints GT (-SD-)')            # GT  fingerprints      (SD-order)
    axs[row_idx][7].set(title='act_gt_SD (Ordered without rep)')   # GT  activity traces   (SD-order)
    axs[row_idx][8].set(title='fingerprints NMF (-SD-)')           # NMF fingeprints       (SD-order)
    axs[row_idx][9].set(title='act_nmf_SD (Ordered without rep)')  # NMF activity traces   (SD-order)

[END, END2] = [datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
print('Analys: A'+ start + ' began at: ' + start2 + ' and finished at: ' + END2)

#%% #################################################################################################################### %%#
#%% [S.00]:  Saving the results (Analysis Log = Alog)
''' -------------------------------------------------------------------------------------------------------------------- '''
# do_saving = True
# do_saving = False

if do_saving is True:
    print('Saving results...')
    # Creating the str 
    Analysis_Start = str('A'+ start)
    Analysis_End = str('A' + END)
    # Defining where to save
    Alog_root_path = Analysislog_root_path
    Alog_parent_folder = Analysislog_folder
    # Alog_parent_folder = '2023.05.14 - Analysis - NMF replicas, bin results'
    Alog_new_folder = Analysis_Start        # It will create a folder named 'Ayyyymmdd_hhmmss'
    Alog_full_path = Alog_root_path + Alog_parent_folder + Analysis_Start
    os.mkdir(str(Alog_full_path))
    # Defining the Alog filename    
    Alog_filename = str(Analysis_Start + '_' + data_name + "_results.h5")
    # Concatenating Path_filename
    Alog_file_str = str(Alog_full_path + '\\' +  Alog_filename)
    sample = NMF_data_str
    # Alog_file_str = str(Analysis_Start + '_' + data_name + "_results.h5")    
    with h5py.File(Alog_file_str, mode = 'w') as f:
        #Saving the information about the sample
        f['sample']             = sample 
        f['data_root_path']     = data_root_path
        f['data_folder']        = data_folder
        f['data_name']          = data_name
        
        # Saving the preprocessing information
        f['do_cropping']            = do_cropping
        f['do_binning']             = do_binning
        f['do_envelop_filtering']   = do_envelop_filtering
        f['binning']                = BINNING
        f['typycal_size']           = typical_size
        f['gaussian']               = gaussian
        
        # Saving the results of the NMF before 1st ordering
        f['num_beads']           = numbeads
        f['rank used']           = rank
        f['act']                 = act
        f['act_gt']              = act_gt
        f['act_gt_norm']         = act_gt_norm
        f['act_nmf_norm']        = act_norm
        f['fingerprints']        = fingerprints
        f['fingerprints_gt']     = fingerprints_gt
        f['video_binned']        = video_binned
        f['video_high']          = video_highpass

    
        # Saving the parameters used in the NMF
        f['model_n_components'] = model_n_components 
        f['model_init']		     = model_init 
        f['model_random_state'] = model_random_state
        f['model_max_iter']	  = model_max_iter 
        f['model_solver']		 = model_solver 
        f['model_l1_ratio']	 = model_l1_ratio 
        f['model_beta_loss']	 = model_beta_loss 
        f['model_verbose']		 = model_verbose
        f['model_alpha_W']		 = model_alpha_W 
        f['model_alpha_H']		 = model_alpha_H 
        
        # Saving the results of the NMF with 1st ordering (Allowing repetition)
        f['act_ordered_1st_order'] = act_ordered
        f['fingerprints_1st_order']= fingerprints_ordered
        
        f['ordering_GT_fullsize']   = ordering_GTidx_fullsize
        f['ordering_NMF_fullsize']  = ordering_NMFidx_fullsize   
        
        f['ordering_GT']            = ordering_GTidx 
        f['ordering_NMF']           = ordering_NMFidx
        f['ordering_1st']           = ordering_1st

        # Saving the results of the NMF with Hierarchical SD-order (without repetition)
        f['act_GTtraces_SD_order']   = act_gt_sd  
        f['act_NMFtraces_SD_order']  = act_nmf_sd        
        f['SD_order_GT']             = SD_order_GT                   
        f['SD_order_NMF']            = SD_order_NMF
        f['SD_order_GT_fullsize']   = SD_order_GT_fullsize
        f['SD_order_NMF_fullsize']       = SD_order_NMF_fullsize
        f['fingerprints_GT_SD_order']   = fingerprints_gt_sd  
        f['fingerprints_NMF_SD_order']  = fingerprints_nmf_sd 
        
        # Saving the results of the NMF with Inversed SD-order (without repetition)
        f['I_order_NMF']            = I_order_NMF           
        f['I_order_GT']             = I_order_GT            
        # f['act_GTtraces_I_order']   = act_gt_inv  
        # f['act_NMFtraces_I_order']  = act_nmf_inv 
        
        # Saving the results of the Correlation Coeficients
        f['corr_r1']                = r1             
        f['corr_GT_GT_sd']          = corr_GT_GT_sd         
        f['corr_NMF_NMF_sd']        = corr_NMF_NMF_sd       
        f['corr_GT_NMF_sd']         = corr_GT_NMF_sd        
        f['corr_NMF_GT_sd']         = corr_NMF_GT_sd  
        f['SD_diagonal']            = SD_diagonal
        
        # Saving information about the Data Analysis:
        f['Analysis_Start']         =  Analysis_Start 
        f['Analysis_End']           =  Analysis_End
        
        # Saving the version of the module:
        f['Module']                 = ModulePy
        
    print('Done!')
    print(' ')
else:
    print('You chose to not save the analysis! ')
    
print(' ')    

#%% Plotting all the time traces right at the output of NMF (without any sorting)
tt_index = 0 #time-trace index to be plotted

fig,ax = plt.subplots(figsize = (8,12))
# ax.set_prop_cycle('color', colors)
SPACING = 1 #spacing between pairs of traces

for idx in range(rank):
    ax.plot(act_norm[:,rank-1-idx] + idx * SPACING + 1, label = 'NMF trace #' + str(idx + 1))
        
y_pxlsticks = np.arange(0,20,1)
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_yticks(np.arange(1,rank+1,1))
plt.suptitle('Individual time trace')

# plt.legend()
plt.tight_layout()
plt.show()

#%% Getting all big NMF images of each Pattern (P)

nrows = 1       
ncols = 1
all_cols = ncols
for pattern in range(numbeads):
    fig_P, axs_P = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(20, 20))
    # im_P = axs_P.imshow(fingerprints_gt_sd[:,:,pattern])
    im_P = axs_P.imshow(fingerprints_nmf_sd[:,:,pattern])   #NMF patterns
    fig_P.colorbar(im_P, shrink=0.825)
    


#%% displaying a white image in the end of all figures - just to separate the output figures
fig_P, axs_P = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(20, 20))


#%% Getting all big GT images of each Pattern (P)

nrows = 1       
ncols = 1
all_cols = ncols
for pattern in range(numbeads):
    fig_P2, axs_P2 = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(20, 20))
    im_P2 = axs_P2.imshow(fingerprints_gt_sd[:,:,pattern])   # GT patterns
    # im_P2 = axs_P2.imshow(fingerprints_nmf_sd[:,:,pattern])
    fig_P2.colorbar(im_P2, shrink=0.825)
    
#%% Displaying a white image in the end of all figures - just to separate the output figures
fig_P, axs_P = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(20, 20))


#%% Displaying each NMF fingerprint to get all the replicas (extra ones) --- Note, here the NMF data before ordering
nrows = 1       
ncols = 1
all_cols = ncols
for pattern in range(rank):
    fig_P, axs_P = plt.subplots(nrows= nrows, ncols=all_cols, figsize=(20, 20))
    im_P = axs_P.imshow(fingerprints[:,:,pattern])
    fig_P.colorbar(im_P, shrink=0.825)


#%% Version log ###################################################################################################################

'''
Created on Thu Aug  4 14:29:43 2022 (Fernando)  - NMF_example.py
Adapted on Tue Jan 31 15:44:50 2023 (Caio)      - NMF_example_adapted.py
Adapted on Fri Feb 10 20:17:12 2023 (Caio)      - NMF_example_adapted_ordertest_v3.py
Adapted on Sun Feb 12 17:38:08 2023 (Caio)      - NMF_scattering_microendoscopy_v8_20230212.py -  works for rank == numbeads
Adapted on Wed Feb 22 15:40:28 2023 (Caio)      - NMF_scattering_microendoscopy_v10_20230222.py - works for rank >= numbeads
Adapted on Fri Feb 24 18:35:16 2023 (Caio)      - NMF_scattering_microendoscopy_v12_20230224.py - works for any rank value
Adapted on Wed Mar  8 10:26:56 2023 (Caio)      - NMF_scattering_microendoscopy_v13_20230308.py - plus, can remove initial temporal frames
Adapted on Mon Mar 13 13:51:41 2023 (Caio)      - NMF_scattering_microendoscopy_v14_20230313.py - plus, manually chooses to normalize traces with max ou mean value
Adapted on Wed Mar 15 11:06:29 2023 (Caio)      - NMF_scattering_microendoscopy_v15_20230315.py - plus, new temporal traces display
Adapted on Thu Mar 16 10:33:08 2023 (Caio)      - NMF_scattering_microendoscopy_v16_20230316.py - plus, getting manually GT fingrinprints from continuous experiment
Adapted on Sat Mar 18 14:59:14 2023 (Caio)      - NMF_scattering_microendoscopy_v17_20230318.py - plus, compacting code through OOP - tools_caio.py (not completed)
Adapted on Sun May 14 18:40:55 2023 (Caio)      - NMF_scattering_microendoscopy_v18_20230514.py - plus, extra optional plots in the end (individual fingerprints and traces)
Adapted on Wed Oct 11 14:53:11 2023 (Caio)      - NMF_scattering_microendoscopy_v19_20231011.py - adapted for multiple sources (chatGPT suggestion)
Adapted on Sat Oct 28 16:50:39 2023 (Caio)      - NMF_scattering_microendoscopy_v20_20231011.py - plus, possibility to remove a single frame from the raw data
Adapted on Fri Jun  7 01:05:06 2024 (Caio)      - NMF_scattering_microendoscopy_v21_20240606.py - plus, add the possibility to chose to include a red trace in the worst trace or not
Adapted on Fri Jun 14 23:26:34 2024 (Caio)      - NMF_demixing__ProofOfPrinciple_v22_20240614.py - Code to share (Commented and organized)
'''
