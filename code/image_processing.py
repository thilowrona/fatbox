import numpy as np
import cv2



## THRESHOLDING
def simple_threshold(arr, value):
    threshold = np.where(arr > value, 1, 0)
    threshold = np.uint8(threshold)
    return threshold

def adaptive_threshold(arr):    
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    arr = 255 * arr
    image = cv2.resize(arr.astype('uint8'), dsize=arr.shape)
    _,threshold = cv2.threshold(image,
                                0,
                                1,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU
                                )
    return threshold



## SKELETONIZE
def guo_hall(arr):
    import cv_algorithms
    skeleton = cv_algorithms.guo_hall(arr)
    skeleton[0, :] = skeleton[1, :]
    skeleton[-1,:] = skeleton[-2,:]
    skeleton[:, 0] = skeleton[:, 1]
    skeleton[:,-1] = skeleton[:,-2]
    return skeleton 