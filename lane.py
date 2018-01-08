import os
import glob
import cv2
import numpy as np
import pickle
from moviepy.editor import VideoFileClip

# Camera calibration coefficients
mtx = None
dist = None

# Perspective Transform
M = None

# Hyperparameters
S_THRESHOLD = (170, 255)
SX_THRESHOLD = (20, 100)
TRANSFORM_SRC_POINT = np.float32([[609,440],[674,440],[224,719],[1115,719]])
TRANSFORM_DEST_POINT = np.float32([[224,0],[1115,0],[224,719],[1115,719]])

def save_image(image, filename, suffix):
    filename.replace('\\', '/')
    splitted_folder_file = filename.split('/')
    splitted_file_ext = splitted_folder_file[1].split('.')
    write_name = splitted_folder_file[0] + '/' + splitted_file_ext[0] + '_' + suffix + '.' + splitted_file_ext[1]
    cv2.imwrite(write_name, image)

def calibrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal\calibration*.jpg')

    # Step through the list and search for chessboard corners
    for filename in images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw the corners and save
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            save_image(img, filename, 'corners_found')
    
    # Do camera calibration given object points and image points
    test_img = cv2.imread('camera_cal/calibration2.jpg')
    img_size = (test_img.shape[1], test_img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Undistort the test image
    dst = cv2.undistort(test_img, mtx, dist, None, mtx)
    cv2.imwrite('camera_cal/calibration2_undistorted.jpg',dst)

    # Save the camera calibration result for later use
    camera_calibration = {}
    camera_calibration["mtx"] = mtx
    camera_calibration["dist"] = dist
    with open('camera_calibration.pickle', 'wb') as pickle_file:
        pickle.dump(camera_calibration, pickle_file)

def binary_threshold(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(image)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 255
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 255) | (sxbinary == 255)] = 255

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((s_binary, sxbinary, np.zeros_like(sxbinary)))
    
    return combined_binary

def detect_lane(image, image_path=None):
    # Undistort the image
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    if image_path is not None:
        save_image(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR), image_path, 'undistorted')

    # line1 = ((292, 670), (609, 440))
    # line2 = ((1038, 670), (674, 440))
    # cv2.line(dst, (int(((719-line1[0][1])*(line1[1][0]-line1[0][0])/(line1[1][1]-line1[0][1]))+line1[0][0]), 719), line1[1], [255, 0, 0], 2)
    # print(int(((719-line1[0][1])*(line1[1][0]-line1[0][0])/(line1[1][1]-line1[0][1]))+line1[0][0]))
    # cv2.line(dst, (int(((719-line2[0][1])*(line2[1][0]-line2[0][0])/(line2[1][1]-line2[0][1]))+line2[0][0]), 719), line2[1], [255, 0, 0], 2)
    # print(int(((719-line2[0][1])*(line2[1][0]-line2[0][0])/(line2[1][1]-line2[0][1]))+line2[0][0]))
    # save_image(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR), img_path, 'line')
    
    # Do binary thresholding to the image
    bin_thresh = binary_threshold(dst, S_THRESHOLD, SX_THRESHOLD)
    if image_path is not None:
        save_image(bin_thresh, image_path, 'binary_threshold')
    
    # Perspective transform into a bird's eye view
    persp_trans = cv2.warpPerspective(dst, M, (dst.shape[1],dst.shape[0]), flags=cv2.INTER_LINEAR)
    if image_path is not None:
        save_image(cv2.cvtColor(persp_trans, cv2.COLOR_RGB2BGR), image_path, 'perspective_transform')
        
    return image

if __name__ == '__main__':
    # # calibrate camera
    # calibrate_camera()

    # Load the camera calibration coefficients
    with open('camera_calibration.pickle', 'rb') as pickle_file:
        camera_calibration = pickle.load(pickle_file)
        mtx = camera_calibration["mtx"]
        dist = camera_calibration["dist"]

    # get the perspective transform matrix
    M = cv2.getPerspectiveTransform(TRANSFORM_SRC_POINT, TRANSFORM_DEST_POINT)

    # # process test images
    # test_images = list(map(lambda s: 'test_images/' + s, os.listdir('test_images/')))
    # for image_path in test_images:
    #     # Read the image
    #     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
    #     # Run the image through the pipeline
    #     detected_lane_image = detect_lane(image)
        
    #     # Save the image
    #     cv2.imwrite(image_path.replace('test_images','test_images_output').replace('.jpg', '')+'_detected.jpg', cv2.cvtColor(detected_lane_image, cv2.COLOR_RGB2BGR))

    # # process test video(s)
    # video = VideoFileClip("project_video.mp4")
    # detected_lane_video = video.fl_image(detect_lane) #NOTE: this function expects color images!!
    # detected_lane_video.write_videofile("project_video_detected.mp4", audio=False)

    img_path = 'test_images/test5.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detect_lane(img, img_path)
