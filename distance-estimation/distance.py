import cv2
import numpy as np


def compute_disparity_map(grayL, grayR):
    '''
    This function generates the disparity map
    :param grayL: [numpy.ndarray] Grayed left image
    :param grayR: [numpy.ndarray] Grayed right image
    :return:
    '''
    print("Generating disparity map ... ")
    # Create StereoSGBM and prepare all parameters
    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    # Used for the filtered image
    stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.8

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
    dispL = disp
    dispR = stereoR.compute(grayR, grayL)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    disp = ((disp.astype(
        np.float32) / 16) - min_disp) / num_disp  # Calculation allowing us to have 0 for the most distant object able to detect

    print("Disparity maps are ready")
    return filteredImg, disp


def reproject_to_3d(disparity_map, Q):
    '''
    This function is one way to calculate the depth.
    The function reproject the disparity map to 3d map, and return it
    '''
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    return depthMap


def compute_distance(disp, bbox):
    '''
    Estimate the distance
    :param disp: [numpy.ndarray] disparity map
    :param bbox: [list] bounding box (x_min, y_min, x_max, y_max)
    :return:
    '''
    h = int((bbox[1] + bbox[3]) / 2)
    w = int((bbox[0] + bbox[2]) / 2)
    average = 0
    count = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[h + u, w + v]
            count += 1
    average = average / 9
    Distance = -593 * average ** (3) + 1500 * average ** (2) - 1385 * average + 522.06
    Distance = np.around(Distance * 0.01, decimals=2)
    # print('Distance: ' + str(Distance * 1000) + ' mm')
    return Distance * 1000
