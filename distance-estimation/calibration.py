import cv2
from io import BytesIO
import numpy as np
import PIL.Image as Image
import PIL.ImageEnhance as ImageEnhance


def pre_process_img(img):
    '''
    This function can pre-process the input image.
    The input image must be PIL.JpegImagePlugin.JpegImageFile type.
    '''
    # Enhance the brightness of the image
    img_enhancer = ImageEnhance.Brightness(img)
    img = img_enhancer.enhance(2)
    img = np.array(img)

    # Gray the image
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


def read_images(left_dataset, right_dataset):
    raw_left = eval(left_dataset['data'][0])
    img_left = Image.open(BytesIO(raw_left))
    raw_right = eval(right_dataset['data'][0])
    img_right = Image.open(BytesIO(raw_right))

    # Pre-process the images
    gray_left = pre_process_img(img_left)
    gray_right = pre_process_img(img_right)

    return gray_left, gray_right


def camera_calibration_no_chess(gray_left, gray_right, K_left):
    image_size = gray_left.shape

    # Termination criteria
    criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("SIFT detecting and matching ... ")
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray_left, None)
    kp2, desc2 = sift.detectAndCompute(gray_right, None)
    bf = cv2.BFMatcher(crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.1 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    print("Generating the essential matrix ... ")
    E, mask = cv2.findEssentialMat(pts1, pts2, K_left,
                                   method=cv2.FM_RANSAC, prob=0.99,
                                   threshold=0.4, mask=None)
    print("E = ", E)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    points,R,t,mask = cv2.recoverPose(E,pts1,pts2,R = None,t = None,mask = None)
    K_inv = np.linalg.inv(K_left)
    F = K_inv.T @ E @ K_inv

    return R, t, E, F


def camera_calibration_chess(left_dataset, right_dataset, K_left, D_left, K_right, D_right):
    '''
    This function performs camera calibration. The calibration algorithm is following: \n
    a) Read images from two input datasets. \n
    b) Detect the chessboard corners. \n
    c) Refine the corner positions. \n
    d) Compute intrinsic and extrinsic params. \n
    :param left_dataset: [pandas.core.frame.DataFrame] dataset stores left images
    :param right_dataset: [pandas.core.frame.DataFrame] dataset stores right images
    :param K_left: [numpy.ndarray] camera matrix of left camera
    :param D_left: [numpy.ndarray] distortion_coefficients of left camera
    :param K_right: [numpy.ndarray] camera matrix of right camera
    :param D_right: [numpy.ndarray] distortion_coefficients of right camera
    :return: R, T, E, F: [numpy.ndarray] camera extrinsic params
    '''
    print("Start to calibrate the camera ...")

    # Setup constant
    length = min(len(left_dataset), len(right_dataset))
    CHECKERBOARD_SIZE = (10, 7) # rol and col of the corners of the chessboard

    # Termination criteria
    criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((np.prod(CHECKERBOARD_SIZE), 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints= []     # 3d points in real world space
    imgpointsR= []    # 2d points in right image plane
    imgpointsL= []    # 2d points in left image plane

    print("Reading ", length, " images ...")
    for i in range(0, length):
        print("Reading No.", i+1, " image now ...")

        # Read left and right images
        raw_left = eval(left_dataset['data'][i])
        img_left = Image.open(BytesIO(raw_left))
        raw_right = eval(right_dataset['data'][i])
        img_right = Image.open(BytesIO(raw_right))

        # Pre-process the images
        gray_left = pre_process_img(img_left)
        gray_right = pre_process_img(img_right)

        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD_SIZE, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD_SIZE, None)

        if ret_left and ret_right:
            # Refine corner locations
            cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # Add to image points lists
            imgpointsL.append(corners_left)
            imgpointsR.append(corners_right)
            objpoints.append(objp)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                                                     imgpointsL, imgpointsR,
                                                                                                     K_left, D_left,
                                                                                                     K_right, D_right,
                                                                                                     (4096, 3000),
                                                                                                     criteria=criteria_stereo)
    print("Camera is ready to use ... ")
    return R, T, E, F


def rectify_image(left_img, right_img, R, T, K_left, D_left, K_right, D_right):
    '''
    This function can rectify the left and right images.
    :param left_img: [numpy.array]
    :param right_img: [numpy.array]
    :param R: [numpy.array] Rotation matrix of left and right cameras
    :param T: [numpy.array] translation matrix of left and right cameras
    :param K_left: [numpy.ndarray] camera matrix of left camera
    :param D_left: [numpy.ndarray] distortion_coefficients of left camera
    :param K_right: [numpy.ndarray] camera matrix of right camera
    :param D_right: [numpy.ndarray] distortion_coefficients of right camera
    :return:
    '''
    print("Rectifying left and right images ... ")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left,
                                                K_right, D_right,
                                                left_img.shape[:2][::-1],
                                                R, T, 0, (0, 0))

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(K_left, D_left,
                                                         R1, P1,
                                                         left_img.shape[:2][::-1], cv2.CV_16SC2)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(K_right, D_right,
                                                           R2, P2,
                                                           left_img.shape[:2][::-1], cv2.CV_16SC2)

    print("Remapping left and right images ... ")
    left_img_rect = cv2.remap(left_img, map_left_x, map_left_y, interpolation=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT)
    right_img_rect = cv2.remap(right_img, map_right_x, map_right_y, interpolation=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT)

    print("Rectification complete")
    return left_img_rect, right_img_rect, Q