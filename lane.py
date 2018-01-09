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

# Polynomial Fits
left_fit = None
right_fit = None

# Hyperparameters
S_THRESHOLD = (170, 255)
SX_THRESHOLD = (20, 100)
TRANSFORM_SRC_POINT = np.float32([[609,440],[674,440],[229,719],[1119,719]])
TRANSFORM_DEST_RATIO = 0.8

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

def find_lines(image, image_path=None):
    # Input : warped binary thresholded image
    out_img = None
    ploty = None
    left_fitx = None
    right_fitx = None
    radius_of_curvature = None
    line_base_pos = None

    global left_fit
    global right_fit

    if left_fit is None :
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # draw the pixels detected by the sliding window
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # draw the fitted lines
        np.clip(left_fitx, 0, 1279, out=left_fitx)
        np.clip(right_fitx, 0, 1279, out=right_fitx)
        out_img[ploty.astype(int), left_fitx.astype(int)] = [255, 255, 0]
        out_img[ploty.astype(int), right_fitx.astype(int)] = [255, 255, 0]

    else:
        # The polynomial fit is already detected on the last frame
        # It's now much easier to find line pixels!
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the search window area
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Draw the fitted lines
        np.clip(left_fitx, 0, 1279, out=left_fitx)
        np.clip(right_fitx, 0, 1279, out=right_fitx)
        out_img[ploty.astype(int), left_fitx.astype(int)] = [255, 255, 0]
        out_img[ploty.astype(int), right_fitx.astype(int)] = [255, 255, 0]

    return out_img

def detect_lane(image, image_path=None):
    # Undistort the image
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    if image_path is not None:
        save_image(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR), image_path, 'undistorted')
    
    # Do binary thresholding to the image
    bin_thresh = binary_threshold(dst, S_THRESHOLD, SX_THRESHOLD)
    if image_path is not None:
        save_image(bin_thresh, image_path, 'binary_threshold')
    
    # Perspective transform into a bird's eye view
    persp_trans = cv2.warpPerspective(bin_thresh, M, (bin_thresh.shape[1],dst.shape[0]), flags=cv2.INTER_LINEAR)
    if image_path is not None:
        save_image(persp_trans, image_path, 'perspective_transform')
    
    # Find lines
    detected_lines = find_lines(persp_trans)
    if image_path is not None:
        save_image(detected_lines, image_path, 'detected_lines')
        
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
    left_point = 640 - ((640-TRANSFORM_SRC_POINT[2][0])*TRANSFORM_DEST_RATIO)
    right_point = 640 + ((TRANSFORM_SRC_POINT[3][0]-640)*TRANSFORM_DEST_RATIO)
    TRANSFORM_DEST_POINT = np.float32([[left_point,0],[right_point,0],[left_point,719],[right_point,719]])
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

    img_path = 'test_images/test1.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detect_lane(img, img_path)
