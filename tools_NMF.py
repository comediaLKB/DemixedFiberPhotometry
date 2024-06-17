# ModulePy = tools_NMF 
"""
@author: Fernando SOLDEVILLA:   https://github.com/cbasedlf ,  https://github.com/cbasedlf/optsim
         Complex Meedia Optics lab: https://github.com/comediaLKB

Useful functions/classes for light field project and general optics
"""

#%% Import stuff
import numpy as np
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift


#plotting stuff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%

def ft2(g,delta):
    '''
    ft2 performs a discretized version of a Fourier Transform by using DFT

    Parameters
    ----------
    g : input field (sampled discretely) on the spatial domain
    delta : grid spacing spatial domain (length units)

    Returns
    -------
    G : Fourier Transform

    '''
    G = fftshift(fft2(ifftshift(g)))*delta**2
    return G

def ift2(G,delta_f):
    '''
    ift2 performs a discretized version of an Inverse Fourier Transform
    by using DFT

    Parameters
    ----------
    G : input field (sampled discretely) on the frequency domain
    delta_f : grid spacing frequency domain (1/length units)

    Returns
    -------
    g : Inverse Fourier Transform

    '''
    n = G.shape[0]
    g = ifftshift(ifft2(fftshift(G)))*(n*delta_f)**2
    return g

#%% Plotting functions + display/saving

def show_Nimg(hypercube,fig_size=False,colormap='viridis'):
    '''
    show_Nimg generates a grid plot from a set of 2D images.

    Parameters
    ----------
    hypercube : Set of 2D images. Third axis should be the image number
    fig_size : size of the figure 
    colormap : TYPE, optional
        DESCRIPTION. The default is 'viridis'.

    '''
    if fig_size == False:
        fig_size = (8,8)
        pass
    Nimg = hypercube.shape[2]
    nrows = int(np.ceil(np.sqrt(Nimg)))
    fig, ax = plt.subplots(nrows = nrows, ncols = nrows, figsize = fig_size)
    counter = 0
    for rowidx in range(0,nrows):
        for colidx in range(0,nrows):
            if counter < Nimg:
                # plt.subplot(ax[rowidx,colidx])
                im = ax[rowidx,colidx].imshow(hypercube[:,:,counter],
                                              cmap = colormap)
                ax[rowidx,colidx].set_aspect(1)
                divider = make_axes_locatable(ax[rowidx,colidx])
                cax = divider.append_axes('right', size='5%', pad = 0.1)
                fig.colorbar(im, cax = cax, ax = ax[rowidx,colidx])
                counter += 1
                pass
            pass
        pass
    plt.tight_layout()
    plt.show()
    pass


def nparray2png(im_array):
    '''
    nparray2png takes a numPy array and converts it to a
    grayscale image object. Useful for saving results to images

    Parameters
    ----------
    im_array : numpy array representing an image

    Returns
    -------
    img : Image object
    '''
    from PIL import Image
    im = (im_array - np.min(im_array)) / (np.max(im_array) - np.min(im_array)) * 255
    im = im.astype('int8')
    img = Image.fromarray(im, mode = 'L')
    return img

def save_images(imgs, namestring = 'img', leadzeros = 2):
    '''
    save_images stores all  the images from a 3D array (px*px*number_of_images 
    shape) as .png files, or a list of 2D arrays (px*px). Second case (list) 
    can be used to save images with different sizes inside a list

    Parameters
    ----------
    imgs : 3D array with the images stored in the third axis (or list of 
                                                                  2D arrays)
    namestring: string for naming the files ('img' by default)
    leadzeros: number of leading zeros (also for naming the files, 2 by default)

    '''
    if type(imgs) == np.ndarray:
        for idx in range(imgs.shape[2]):
            pic = nparray2png(imgs[:,:,idx])
            pic.save(namestring + '_' + 
                 ('{:0' + str(leadzeros) + 'd}').format(idx) + '.png')
        print('Saving done')
    elif type(imgs) == list:
        for idx in range(len(imgs)):
            pic = nparray2png(imgs[idx])
            pic.save(namestring + '_' +
                  ('{:0' + str(leadzeros) + 'd}').format(idx) + '.png')
        print('Saving done')
    else:
        print(
        'Wrong input format. Provide either 3D nparray or a list of 2D arrays')
    
    return 


def cropROI(img,size,center_pos):
    '''
    cropROI gets a ROI with a desired size, center at a fixed position

    Parameters
    ----------
    img : input image
    size : size of the ROI (2 element vector, size in [rows,cols] format)
    center_pos : central position of the ROI

    Returns
    -------
    cropIMG = cropped ROI of the image

    '''
    if img.shape[0]< size[0] or img.shape[1] < size[1]:
        print('Size is bigger than the image size')
        return img
    else:
        center_row = center_pos[0]
        center_col = center_pos[1]
        semiROIrows = int(size[0]/2)
        semiROIcols = int(size[1]/2)
        cropIMG = img[center_row - semiROIrows : center_row + semiROIrows,
                      center_col - semiROIcols : center_col + semiROIcols]
        pass
    
    return cropIMG


def buildGauss(px,sigma,center,phi):
    """
    buildGauss generates a Gaussian function in 2D. Formula from
    https://en.wikipedia.org/wiki/Gaussian_function

    Parameters
    ----------
    px : image size of the output (in pixels)
    sigma : 2-element vector, sigma_x 
    and sigma_y for the 2D Gaussian
    center : 2-element vector, center position
    of the Gaussian in the image
    phi : Rotation angle for the Gaussian

    Returns
    -------
    gaus : 2D image with the Gaussian

    """
    #Generate mesh
    x = np.linspace(1,px,px)
    X,Y = np.meshgrid(x,x)
    
    #Generate gaussian parameters
    a = np.cos(phi)**2/(2*sigma[0]**2) + np.sin(phi)**2/(2*sigma[1]**2)
    b = -np.sin(2*phi)/(4*sigma[0]**2) + np.sin(2*phi)/(4*sigma[1]**2)
    c = np.sin(phi)**2/(2*sigma[0]**2) + np.cos(phi)**2/(2*sigma[1]**2)
    
    #Generate Gaussian
    gaus = np.exp(-(a*(X-center[0])**2 + 2*b*(X-center[0])*(Y-center[1]) + c*(Y-center[1])**2))
    
    return gaus

def filt_fourier(img,filt_func):
    '''
    filt_fourier filters an image in the Fourier domain.
    To do so, it uses [filt_func]. It multiplies that mask to the 
    Fourier transform of the input image [img], thus eliminating some 
    frequency content. Then it goes back to image domain.

    Parameters
    ----------
    img : Input image (to be filtered)
    filt_func : Filtering mask in the Fourier domain
    
    Returns
    -------
    img_filt : Filtered image

    '''
    # Go to Fourier domain
    img_k = fftshift(fft2(fftshift(img)))
    # Apply filter
    img_k_filt = img_k*filt_func
    # Go back to image domain
    img_filt = np.abs(ifftshift(ifft2(ifftshift(img_k_filt))))
    return img_filt


def iseven(number):
    '''
    Should be pretty easy to see what this does

    Parameters
    ----------
    number : input number
    Returns
    -------
    True/False if the number is even/odd

    '''    
    return number % 2 == 0

def isodd(number):
    '''
    Should be pretty easy to see what this does

    Parameters
    ----------
    number : input number
    Returns
    -------
    True/False if the number is odd/even

    '''    
    return number % 2 != 0

