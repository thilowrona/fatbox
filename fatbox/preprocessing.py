# Packages
import cv2
import numpy as np

# Funcktions
from skimage.morphology import skeletonize

# Custom package
# import cv_algorithms



#==============================================================================
# This file contains a series of functions to process an array before extrac-
# ting faults. This includes functions for: 
# (1) thresholding
# (2) skeletonize
# (3) labelling connected components
# (4) removing components
# (5) conversion to points. 
#==============================================================================



#******************************************************************************
# (1) THRESHOLDING
# A couple of functions that allow you to threshold your data in different ways
#******************************************************************************

def simple_threshold_binary(arr, threshold):
    """ Thresholds array into a binary array
    
    Parameters
    ----------
    arr : np.array
        Input array that we binarize with threshold
    
    threshold : int, float
        The threshold used to binarize the input array
    
    Returns
    -------  
    arr
        Binarized output array (type: uint8)
    """
    
    # Assertions
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    assert isinstance(threshold, int) or isinstance(threshold, float), "Threshold is neither int nor float"
    
    # Calculation
    arr = np.where(arr > threshold, 1, 0)
    arr = np.uint8(arr)    
    
    return arr




def adaptive_threshold(arr):
    """ Thresholds array into a binary array using an adaptive threshold (Binary+Otsu)
    
    Parameters
    ----------
    
    arr : np.array
        Input array that we binarize with threshold
    
    Returns
    -------  
    arr
        Binarized output array (type: uint8)
    """    
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"    

    # Calculation
    # Scale to [0,1]    
    arr = (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))
    # Scale to [0,255]
    arr = 255 * arr
    # Create image
    image = cv2.resize(arr.astype('uint8'), dsize=(arr.shape[1], arr.shape[0]))
    # Apply adaptive threshold
    _, arr = cv2.threshold(image,
                           0,
                           1,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Convert back to NumPy array
    arr = np.uint8(arr) 
    
    return arr




#******************************************************************************
# (2) SKELETONIZE
# A couple of functions that allow you to skeletonize your data, i.e. reduce to
# one pixel thick lines
#******************************************************************************

def skeleton_scipy(arr):
    """ Basic skeletonize function from SciPy
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    
    Returns
    -------  
    arr
        Output array
    """
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"    
    
    return skeletonize(arr)



def skeleton_guo_hall(arr):
    """ Optimized skeletonize function from cv_algorithms (https://github.com/ulikoehler/cv_algorithms)
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    
    Returns
    -------  
    arr
        Output array
    """
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array" 
    
    # Calculation
    arr = cv_algorithms.guo_hall(arr)
    
    # Correct edge effect
    arr[0, :] = arr[1, :]
    arr[-1,:] = arr[-2,:]
    arr[:, 0] = arr[:, 1]
    arr[:,-1] = arr[:,-2]
    
    return arr




#******************************************************************************
# (3) CONNECTED COMPONENTS
# A function to label connected components in array
#******************************************************************************

def connected_components(arr):
    """ Label connected components
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    
    Returns
    -------  
    ret
        Output array
    markers
        Components
    """
    
    # Assertion
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    
    # Calculation
    ret, markers = cv2.connectedComponents(arr)
    
    return ret, markers




#******************************************************************************
# (4) REMOVAL
# A couple of functions to remove certain components
#******************************************************************************

def remove_small_regions(arr, size):
    """ Remove components below certain size
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    size : int
    
    Returns
    -------  
    arr
        Output array
    """
    # Assertion
    assert isinstance(arr, np.ndarray), "Input array is not a NumPy array"
    assert isinstance(arr, int), "Input size is not an integer "
    
    
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    
    # connectedComponentswithStats yields every seperated component with
    # information on each of them, such as size
    # the following part is just taking out the background which is also
    # considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of
    # the sizes or whatever

    # your answer image
    arr = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= size:
            arr[output == i + 1] = 255

    # Convert to uint8
    arr = np.uint8(arr)    

    return arr




def remove_large_regions(arr, size):
    """ Remove components above certain size
    
    Parameters
    ----------
    
    arr : np.array
        Input array
    size : int
    
    Returns
    -------  
    arr
        Output array
    """
    # Assertion
    assert isinstance(arr, np.ndarray), "Input array is not a NumPy array"
    assert isinstance(arr, int), "Input size is not an integer "    
    
    
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    
    # connectedComponentswithStats yields every seperated component with
    # information on each of them, such as size
    # the following part is just taking out the background which is also
    # considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of
    # the sizes or whatever

    # your answer image
    arr = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] <= size:
            arr[output == i + 1] = 255

    # Convert to uint8
    arr = np.uint8(arr)    

    return arr




#******************************************************************************
# (5) CONVERSION
# A function to convert an array to points (x,y)
#******************************************************************************

def array_to_points(arr):
    """ A function to convert an array to points (x,y)
    
    Parameters
    ----------
    arr : np.array
        Input array that we binarize with threshold
    
    
    Returns
    -------  
    arr
        Output array (points)
    """
    
    # Assertions
    assert isinstance(arr, np.ndarray), "Input is not a NumPy array"
    
    # Calculation
    n = np.count_nonzero(arr)
    points = np.zeros((n, 2))
    (points[:, 1], points[:, 0]) = np.where(arr != 0)
    
    return points




