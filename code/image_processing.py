import numpy as np
import cv2



## THRESHOLDING
def simple_threshold_binary(arr, value):
    threshold = np.where(arr > value, 1, 0)
    threshold = np.uint8(threshold)
    return threshold

def simple_threshold_truevalue(arr, value):
    threshold = np.where(arr > value, arr, 0)
    return threshold


def adaptive_threshold(arr):    
    arr = (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))
    arr = 255 * arr
    image = cv2.resize(arr.astype('uint8'), dsize=(arr.shape[1], arr.shape[0]))
    _,threshold = cv2.threshold(image,
                                0,
                                1,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU
                                )
    return threshold





## FILTERING
def remove_small_regions(arr, size):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    
    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= size:
            img2[output == i + 1] = 255
    
    return np.uint8(img2)


def remove_large_regions(arr, size):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    
    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] <= size:
            img2[output == i + 1] = 255
    
    return np.uint8(img2)
















## SKELETONIZE
def guo_hall(arr):
    import cv_algorithms
    skeleton = cv_algorithms.guo_hall(arr)
    skeleton[0, :] = skeleton[1, :]
    skeleton[-1,:] = skeleton[-2,:]
    skeleton[:, 0] = skeleton[:, 1]
    skeleton[:,-1] = skeleton[:,-2]
    return skeleton 



# CONVERT TO POINTS
def np_to_points(arr):
    N = np.count_nonzero(arr)
    points = np.zeros((N, 2))
    (points[:,0], points[:,1]) = np.where(arr != 0)
    return points




## ADD COORDINATES
def add_coordinates(points, x, y, z):
    N = points.shape[0]    
    points_new = np.zeros((N,5))
    points_new[:,:2] = points
    for n, row in enumerate(points):
        points_new[n,2] = x[int(row[1])]
        points_new[n,3] = y
        points_new[n,4] = z[int(row[0])]
    return points_new