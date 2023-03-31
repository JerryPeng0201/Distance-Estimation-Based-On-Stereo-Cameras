from bagpy import bagreader
import yaml
import pandas as pd
import argparse
import time
from distance import *
from calibration import *

if __name__ == "__main__":
    start_time = time.time()
    # create the ArgumentParser object
    parser = argparse.ArgumentParser(description='This is a script that does something.')

    # add arguments to the parser
    parser.add_argument('bagfile', type=str, help='.bag file that stores the dataset')
    parser.add_argument('left_camera', type=str, help='YAML file for left camera')
    parser.add_argument('right_camera', type=str, help='YAML file for right camera')
    parser.add_argument('-l', '--list', nargs='+', type=int, help='<Required> bounding box of target area')
    parser.add_argument('-c', '--chess', type=str, help='<Required> is there any chessboard in the pic?')

    # parse the command-line arguments
    args = parser.parse_args()
    dataset_address = args.bagfile
    left_camera = args.left_camera
    right_camera = args.right_camera
    bbox = args.list
    chess = args.chess

    # Load the dataset
    b = bagreader(dataset_address)
    left_data = b.message_by_topic('/left/image_raw/compressed')
    df_left_image = pd.read_csv(left_data)

    right_data = b.message_by_topic('/right/image_raw/compressed')
    df_right_image = pd.read_csv(right_data)

    with open(left_camera, 'r') as stream:
        try:
            left_camera_loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open(right_camera, 'r') as stream:
        try:
            right_camera_loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    K_left = np.array(left_camera_loaded['camera_matrix']['data']).reshape(3, 3)
    D_left = np.array(left_camera_loaded['distortion_coefficients']['data'])
    P_left = np.array(left_camera_loaded['projection_matrix']['data']).reshape(3, 4)
    R_left = np.array(left_camera_loaded['rectification_matrix']['data']).reshape(3, 3)

    K_right = np.array(right_camera_loaded['camera_matrix']['data']).reshape(3, 3)
    D_right = np.array(right_camera_loaded['distortion_coefficients']['data'])
    P_right = np.array(right_camera_loaded['projection_matrix']['data']).reshape(3, 4)
    R_right = np.array(right_camera_loaded['rectification_matrix']['data']).reshape(3, 3)

    # Camera Calibration
    if chess == 'y':
        R, T, E, F = camera_calibration_chess(df_left_image, df_right_image, K_left, D_left, K_right, D_right)
    else:
        R, T, E, F = camera_calibration_no_chess(df_left_image, df_right_image, K_left, D_left, K_right, D_right)
    print("R = ", R)
    print("")
    print("T = ", T)
    print("")

    # Read the images
    raw_left = eval(df_left_image['data'][10])
    left_img = Image.open(BytesIO(raw_left))
    left_img = np.array(left_img)  # pre_process_img(left_img)

    raw_right = eval(df_right_image['data'][10])
    right_img = Image.open(BytesIO(raw_right))
    right_img = np.array(right_img)  # pre_process_img(right_img)

    # Rectificate images
    left_img_rect, right_img_rect, Q = rectify_image(left_img, right_img, R, T, K_left, D_left, K_right, D_right)

    # Compute disparity map
    filteredImg, disp = compute_disparity_map(left_img_rect, right_img_rect)

    # Distance estimation
    distance = compute_distance(disp, bbox)
    print('Distance: '+ str(distance)+' mm')
    end_time = time.time()
    duration = end_time - start_time
    print('Execution time: ', duration)