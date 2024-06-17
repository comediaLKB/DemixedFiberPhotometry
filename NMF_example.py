# -*- coding: utf-8 -*-
ModulePy= str('NMF_example.py') 
"""
Description: simple code (example) for NMF demixing used in fluorescence speckles (fully evolved speckles) written by Fernando SOLDEVILA.

- Input files: 
(i) raw_ensemble_video_data.mat                      [videodata_ddmmyyyy_001.mat]
(ii) ground_truth_patterns_traces_gt.mat             [videodata_ddmmyyyy_001_gt.mat]
(iii) tools_fer.py 

----------------------------------------------------------------------------------------------
Created on Thu Aug  4 14:29:43 2022

@author: Fernando SOLDEVILA: https://github.com/cbasedlf , https://github.com/cbasedlf/optsim
         Complex Media Optics lab: https://github.com/comediaLKB
----------------------------------------------------------------------------------------------
"""
### CODE STEPS ###
''' 
Import dataset (from Matlab)
Clean it a little bit (do binning, filter background, etc)
Perform NMF on the video
Take a look at the results and compare with ground truth
'''
#%% ###################################################################### %%#
#%% Import libraries, etc.
# import sys
# sys.path.append('../NMF_lightfield/localization-shackhartmann')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
import scipy.signal
from skimage import restoration
from PIL import Image
from sklearn.decomposition import NMF
import tools_NMF as tNMF

#%% ###################################################################### %%#
#%% Load dataset & ground truth activations
#load ground truth
# fgt = sio.loadmat('data_16112021_005_gt.mat')
fgt = sio.loadmat('data_09112022_001_gt.mat')
act_gt = fgt['pat'].T
#normalize ground truth traces (for doing comparisons later)
act_gt_norm = tNMF.norm_dimension(act_gt, dim = 1, mode='0to1')
#load dataset
# f = h5py.File('data_16112021_005.mat', mode = 'r')
f = h5py.File('data_09112022_001.mat', mode = 'r')
video = np.array(f['video_data'])
video = np.moveaxis(video,0,2) #rearrange to px*px*time form

#%% ###################################################################### %%#
#%% Clean dataset: binning (to go faster) + filtering

#Crop central part of the images
frame_size = np.asarray((720,620), dtype = int)
#check orientation of the frame (vertical video or horizontal)
if frame_size[0] > frame_size[1]:
    ORIENTATION = 'vertical'
elif frame_size[0] < frame_size[1]:
    ORIENTATION = 'horizontal'
else:
    ORIENTATION = 'square'

#preallocate cropped video
video_small = np.zeros((frame_size[0], frame_size[1], video.shape[2]),
                       dtype = 'uint16')
#crop the video
for idx in range(video.shape[2]):
    video_small[:,:,idx] = tNMF.cropROI(video[:,:,idx], size = frame_size,
                                        center_pos = (int(frame_size[0]/2),
                                                    int(frame_size[1]/2)))
    
    
#Do binning
BINNING = 3 #bin size
#calculate new frame sizes after binning
new_frame_size = (frame_size/BINNING).astype(int)
short_side = np.min(new_frame_size)#calculate short side
long_side = np.max(new_frame_size)#calculate long side
#preallocate binned video
video_binned = np.zeros((new_frame_size[0], new_frame_size[1], video.shape[2]))
#do the binning
print('Binning...')
for idx in range(video.shape[2]):
    temp = video_small[:,:,idx]
    temp = Image.fromarray(temp).resize((new_frame_size[1], new_frame_size[0]),
                                        resample = Image.NEAREST)
    temp = np.array(temp)
    video_binned[:,:,idx] = temp
print('done')

### Gaussian test
video_highpass = video_binned

print('Filtering the envelope... ')
#Filter low frequency background (envelope)
video_highpass = np.zeros(video_binned.shape) #preallocate
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
print('End of Envelope filtering ')

#%% ###################################################################### %%#
#%%Do NMF on the pre-processed video (croped,binned, and filtered)

numbeads = act_gt.shape[1] #get real number of beads (from ground truth)
numframes = video.shape[2] - act_gt.shape[1] #get number of frames of the video

#set rank as number of beads + background
rank = numbeads             # rank = exact number of sources
# rank = numbeads + 1       # rank = number of sources + background


#remove ground truth frames (initial part of the acquired video in the lab)
X = video_highpass[:,:,numbeads::].copy()
#reshape intro matrix form
X = np.reshape(X, (new_frame_size[0] * new_frame_size[1], X.shape[2]))

# Do the NMF without a priori knowledge in the initialization init='nndsvd'
 # Create the model:
  # Parameters of the model (that will be saved in the h5py file)  
model_n_components = rank 
model_init = 'nndsvd'
model_random_state = 0
model_max_iter = 3000
model_solver = 'cd'
model_l1_ratio = 0.5
model_beta_loss = 2
model_verbose = 0
model_alpha_W = 1.5
model_alpha_H = 0.5

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

#%% ###################################################################### %%#
#%% Plotting the results of the NMF (traces and fingerprints)
#identifying the recovered traces and plotting them with the ground truth
act = H.T
act_norm = tNMF.norm_dimension(act, dim = 1, mode = '0to1') #normalize recovery

ordering = np.zeros(rank, dtype = 'int') #preallocating
'''
Take each recovered trace, correlate with the ground truth, identify corres_
pondence by looking at the maximum of the correlation. Do that with all the
traces, so in the end you get as many pairs of recovery-ground truth as the
rank of the system.
Sometimes this gives errors (if the traces from the NMF are not very good,
 you might "clasify" 2 traces from the NMF to the same ground truth trace)
'''
r1 = np.zeros((numbeads, rank)) #correlation coeff (preallocating)
for idx in range(numbeads):
    #pick ground truth temporal trace
    temp_gtruth = act_gt_norm[:,idx]
    #Find equivalent eigenvector in NMF reconstruction (normalized)
    for tidx in range(rank):
        #Compute correlation between NMF traces and the current temporal trace
        #Pick element (1,0) of the correlation matrix (correlation between the
        #two vectors)
        r1[idx,tidx] = np.corrcoef(temp_gtruth, act_norm[:,tidx])[1,0]
    #Identify the NMF traces that match the current temporal trace by using
    #the maximum correlation value (this is what sometimes introduces errors.
    #need to think some way to evade it)
    ordering[idx] = int(np.where(r1[idx,:] == np.max(r1[idx,:]))[0])

#re-order spatial fingerprints to match ground truth order, then show
fingerprints_ordered = fingerprints[:,:,ordering].copy()
#show recovered fingerprints
tNMF.show_Nimg(fingerprints_ordered[:,:,0:numbeads])
#show recovered fingerprints
# tNMF.show_Nimg(fingerprints_ordered)

#Swap order to match ground truth ordering
act_ordered = act_norm[:,ordering]
#create colormap (to ease visualization)
colormap = plt.cm.nipy_spectral
colors = [colormap(idx) for idx in np.linspace(0, 1, 4 * numbeads)]
#Plot all traces
fig,ax = plt.subplots(figsize = (8,10))
ax.set_prop_cycle('color', colors)
SPACING = 2 #spacing between pairs of traces
for idx in range(numbeads):
    ax.plot(act_gt[:,idx] + idx * SPACING, label = 'G.T. trace #' + str(idx + 1))
    ax.plot(act_ordered[:,idx] + idx * SPACING + 0.6,
            label = 'NMF trace #' + str(idx + 1))
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Intensity (a.u.)')
plt.suptitle('Temporal activity recovery via NMF')
# plt.legend()
plt.tight_layout()
plt.show()

#Plot all recovered traces (without the ground truth ones for comparison)
fig,ax = plt.subplots(figsize = (8,10))
#create colormap (to ease visualization)
colormap = plt.cm.nipy_spectral
colors = [colormap(idx) for idx in np.linspace(0, 1, rank)]
ax.set_prop_cycle('color', colors)
SPACING = 2 #spacing between pairs of traces
for idx in range(rank):
    # ax.plot(act_gt[:,idx] + idx * SPACING, label = 'G.T. trace #'+ str(idx + 1))
    ax.plot(act_norm[:,idx] + idx * SPACING, label = 'NMF trace #' + str(idx + 1))
ax.set_xlabel('Time (a.u.)')
ax.set_ylabel('Intensity (a.u.)')
plt.suptitle('Temporal activity recovery via NMF')
# plt.legend()
plt.tight_layout()
plt.show()


#%% Saving the results
'''
these are the variables inside the h5 file:
['act', 'act_gt', 'act_norm', 'act_ordered', 'binning', 'fingerprints', 'fingerprints_gt', 'num_beads', 'obj', 'obj_gt', 'video_binned', 'video_high']

activities, activities ground trurh, activities normalized, activities ordered (to match the order of the ground truth ones), binning value, the fingerprints, the fingerprints ground truth, the number of beads, the distribution of beads (spatial locations after doing the localization), the ground truth distribution of beads, the video after binning, the video in high resolution (without binning)
'''
resultsfilename = 'results_09112022_001_rank6_v2.h5'

with h5py.File(resultsfilename, mode = 'w') as f:
    f['act'] = act
    f['act_gt'] = act_gt
    f['act_norm'] = act_norm
    f['act_ordered'] = act_ordered
    f['binning'] = BINNING
    f['fingerprints'] = fingerprints
    f['fingerprints_gt'] = fingerprints_gt
    f['fingerprints_ordered']= fingerprints_ordered
    f['num_beads'] = numbeads
    # f['obj'] = obj
    # f['obj_gt'] = obj_gt
    f['video_binned'] = video_binned
    f['video_high'] = video_highpass

    # Saving the parameters used in the NMF
    f['model_n_components'] = model_n_components 
    f['model_init']		     = model_init 
    f['model_random_state'] = model_random_state
    f['model_max_iter']	     = model_max_iter 
    f['model_solver']		 = model_solver 
    f['model_l1_ratio']		 = model_l1_ratio 
    f['model_beta_loss']	 = model_beta_loss 
    f['model_verbose']		 = model_verbose
    f['model_alpha_W']		 = model_alpha_W 
    f['model_alpha_H']		 = model_alpha_H 
    
#%% Checking the current directory path
import os
os.path.abspath(os.getcwd())

#%%
import math as mt
import numpy as np
int(mt.floor(697/10))*10

#%%
print('1')
