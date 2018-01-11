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
Minv = None

# Polynomial Fits
LEFT_LINE = None
RIGHT_LINE = None
#left_fit = None
#right_fit = None

# Hyperparameters
S_THRESHOLD = (170, 255)
SX_THRESHOLD = (40, 100)
TRANSFORM_SRC_POINT = np.float32([[609,440],[674,440],[229,719],[1119,719]])
TRANSFORM_DEST_RATIO = 0.6
TRANSFORM_DEST_POINT = None

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

def find_lines(image):
    global LEFT_LINE
    global RIGHT_LINE
    # Input : warped binary thresholded image
    # Output : 
    # out_img = None
    # ploty = None
    # left_fitx = None
    # right_fitx = None
    # radius_of_curvature = None
    # car_pos = None

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 90
    # Set minimum number of pixels found to recenter window
    minpix = 100

    def sliding_window(image):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
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
                current = np.int(np.mean(nonzerox[good_left_inds]))
                leftx_dif = leftx_current - current
                leftx_current = current
            if len(good_right_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_right_inds]))
                rightx_dif = rightx_current - current
                rightx_current = current
            # If one window is found, set the other window with the same offset
            if (len(good_left_inds) > minpix) and not(len(good_right_inds) > minpix):
                rightx_current = rightx_current - leftx_dif
            if not(len(good_left_inds) > minpix) and (len(good_right_inds) > minpix):
                leftx_current = leftx_current - rightx_dif

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

        return out_img, ploty, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty

    def fit_window(image, left_fit, right_fit):
        # The polynomial fit is already detected on the last frame
        # It's now much easier to find line pixels!
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
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
        old_left_fit = left_fit
        old_right_fit = right_fit
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        old_left_fitx = old_left_fit[0]*ploty**2 + old_left_fit[1]*ploty + old_left_fit[2]
        old_right_fitx = old_right_fit[0]*ploty**2 + old_right_fit[1]*ploty + old_right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((image, image, image))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([old_left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([old_left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([old_right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([old_right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the search window area
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return out_img, ploty, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty
    
    def draw_calculate(image, ploty, left_fitx, right_fitx):
        # Draw the fitted lines
        left_fitx_clipped = np.empty_like(left_fitx)
        right_fitx_clipped = np.empty_like(right_fitx)
        np.clip(left_fitx, 0, 1279, out=left_fitx_clipped)
        np.clip(right_fitx, 0, 1279, out=right_fitx_clipped)
        out_img[ploty.astype(int), left_fitx_clipped.astype(int)] = [255, 255, 0]
        out_img[ploty.astype(int), right_fitx_clipped.astype(int)] = [255, 255, 0]
        
        # Calculate radius of curvature
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        # meters per pixel in y dimension
        ym_per_pix = 32/720 
        # meters per pixel in x dimension
        xm_per_pix = 3.7/(TRANSFORM_DEST_POINT[1][0] - TRANSFORM_DEST_POINT[0][0])
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_radius_of_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_radius_of_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # left_radius_of_curvature = ((1 + (2*(xm_per_pix/(ym_per_pix)**2)*left_fit[0]*y_eval + (xm_per_pix/ym_per_pix)*left_fit[1])**2)**1.5) / np.absolute(2*(xm_per_pix/(ym_per_pix)**2)*left_fit[0])
        # right_radius_of_curvature = ((1 + (2*(xm_per_pix/(ym_per_pix)**2)*right_fit[0]*y_eval + (xm_per_pix/ym_per_pix)*right_fit[1])**2)**1.5) / np.absolute(2*(xm_per_pix/(ym_per_pix)**2)*right_fit[0])
        radius_of_curvature = (left_radius_of_curvature + right_radius_of_curvature)/2

        # Calculate the car position relative to the center of the lanes
        car_pos = (640 - (right_fitx[719] + left_fitx[719])/2) * xm_per_pix

        return out_img, ploty, left_fitx, right_fitx, radius_of_curvature, car_pos

    def sanity_check(ploty, fit, fitx, x, y, line):
        # if np.sum(np.absolute(left_fitx - left_line.recent_xfitted[-1])) > 1000:
        #     return False
        # if np.sum(np.absolute(right_fitx - right_line.recent_xfitted[-1])) > 1000:
        #     return False

        # left_fit_diff = np.absolute(left_fit - left_line.current_fit)
        # if left_fit_diff[0] > 2:
        #     return False
        # if left_fit_diff[1] > 2:
        #     return False

        # right_fit_diff = np.absolute(right_fit - right_line.current_fit)
        # if right_fit_diff[0] > 2:
        #     return False
        # if right_fit_diff[1] > 2:
        #     return False

        # The A coefficient does not jump wildly
        # if len(line.recent_fit)==5 and abs(fit[0] - sum(list(zip(*line.recent_fit))[0])/5) > 0.00035:
        #     return False
        if not np.any(x[y<360]):
            return False

        return True

    def sanity_check_both(ploty, left_fit, right_fit, left_fitx, right_fitx, left_line, right_line):

        # The distance between the detected lines is reasonable
        if np.absolute(np.absolute(np.mean(left_fitx - right_fitx)) - (TRANSFORM_DEST_POINT[1][0] - TRANSFORM_DEST_POINT[0][0])) > 100:
            return False

        return True


    if LEFT_LINE is None:
        # Create line objects to hold recent data
        LEFT_LINE = Line()
        RIGHT_LINE = Line()

        # Perform sliding window to detect line
        out_img, ploty, left_fit, right_fit, left_fitx, right_fitx, _, _, _, _ = sliding_window(image)

        # Update the line objects
        LEFT_LINE.detected = True
        LEFT_LINE.recent_xfit.append(left_fitx)
        LEFT_LINE.recent_fit.append(left_fit)

        RIGHT_LINE.detected = True
        RIGHT_LINE.recent_xfit.append(right_fitx)
        RIGHT_LINE.recent_fit.append(right_fit)

    else:
        # Perform fit window to detect line
        out_img, ploty, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty = fit_window(image, LEFT_LINE.recent_fit[-1], RIGHT_LINE.recent_fit[-1])
        
        left_sane = sanity_check(ploty, left_fit, left_fitx, leftx, lefty, LEFT_LINE)
        right_sane = sanity_check(ploty, right_fit, right_fitx, rightx, righty, RIGHT_LINE)

        if left_sane and not right_sane:
            right_fit = list(left_fit)
            right_fit[2] += (TRANSFORM_DEST_POINT[1][0] - TRANSFORM_DEST_POINT[0][0])
            RIGHT_LINE.detected = False
            LEFT_LINE.detected = True
        
        if not left_sane and right_sane:
            left_fit = list(right_fit)
            left_fit[2] -= (TRANSFORM_DEST_POINT[1][0] - TRANSFORM_DEST_POINT[0][0])
            LEFT_LINE.detected = False
            RIGHT_LINE.detected = True

        if (not left_sane and not right_sane) or not sanity_check_both(ploty, left_fit, right_fit, left_fitx, right_fitx, LEFT_LINE, RIGHT_LINE):
            # # Perform sliding window to detect line
            # out_img, ploty, left_fit, right_fit, left_fitx, right_fitx = sliding_window(image)
            left_fit = LEFT_LINE.recent_fit[-1]
            right_fit = RIGHT_LINE.recent_fit[-1]
            left_fitx = LEFT_LINE.recent_xfit[-1]
            right_fitx = RIGHT_LINE.recent_xfit[-1]

            LEFT_LINE.detected = False
            RIGHT_LINE.detected = False
        
        else:
            LEFT_LINE.detected = True
            RIGHT_LINE.detected = True

        # Update the line objects
        if len(LEFT_LINE.recent_xfit) == LEFT_LINE.n:
            LEFT_LINE.recent_xfit.pop(0)
        LEFT_LINE.recent_xfit.append(left_fitx)
        if len(LEFT_LINE.recent_fit) == LEFT_LINE.n:
            LEFT_LINE.recent_fit.pop(0)
        LEFT_LINE.recent_fit.append(left_fit)
        
        if len(RIGHT_LINE.recent_xfit) == RIGHT_LINE.n:
            RIGHT_LINE.recent_xfit.pop(0)
        RIGHT_LINE.recent_xfit.append(right_fitx)
        if len(RIGHT_LINE.recent_fit) == RIGHT_LINE.n:
            RIGHT_LINE.recent_fit.pop(0)
        RIGHT_LINE.recent_fit.append(right_fit)
    
    # # Draw the polynomial coefficients
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(out_img,'Left A=' + '{0:.5f}'.format(left_fit[0]), (50,70), font, 1, (255,255,255),1,cv2.LINE_AA)
    # cv2.putText(out_img,'Left B=' + '{0:.5f}'.format(left_fit[1]), (50,120), font, 1, (255,255,255),1,cv2.LINE_AA)
    # cv2.putText(out_img,'Left C=' + '{0:.5f}'.format(left_fit[2]), (50,170), font, 1, (255,255,255),1,cv2.LINE_AA)
    
    # cv2.putText(out_img,'Right A=' + '{0:.5f}'.format(right_fit[0]), (50,220), font, 1, (255,255,255),1,cv2.LINE_AA)
    # cv2.putText(out_img,'Right B=' + '{0:.5f}'.format(right_fit[1]), (50,270), font, 1, (255,255,255),1,cv2.LINE_AA)
    # cv2.putText(out_img,'Right C=' + '{0:.5f}'.format(right_fit[2]), (50,320), font, 1, (255,255,255),1,cv2.LINE_AA)

    return draw_calculate(out_img, ploty, left_fitx, right_fitx)

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # number of history data
        self.n = 5
        # was the line detected in the last iteration?
        self.detected = False  
        #x values for detected line pixels
        #self.allx = None  
        #y values for detected line pixels
        #self.ally = None
        # x values of the last n fits of the line
        self.recent_xfit = [] 

        #polynomial coefficients of the last n fits of the line
        self.recent_fit = [] 
        #difference in polynomial coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 
        #radius of curvature of the line in some units
        #self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None 

def draw_detection(undist, warped, Minv, ploty, left_fitx, right_fitx, radius_of_curvature, car_pos):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    # left_fitx = left_fitx[50:]
    # right_fitx = right_fitx[50:]
    # ploty = ploty[50:]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Draw the radius of curvature
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Radius of Curvature = ' + '{0:.1f}'.format(radius_of_curvature) + 'm', (50,70), font, 2, (255,255,255),2,cv2.LINE_AA)
    # Draw the car position
    cv2.putText(result,'Vehicle is ' + '{0:.1f}'.format(abs(car_pos)) + 'm ' + ('left' if car_pos < 0 else 'right') + ' of center',(50,140), font, 2, (255,255,255),2,cv2.LINE_AA)

    return result

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
    detected_lines, ploty, left_fitx, right_fitx, radius_of_curvature, car_pos = find_lines(persp_trans)
    if image_path is not None:
        save_image(detected_lines, image_path, 'detected_lines')
    
    # Draw the detected lines, lane, radius of curvature, car position into the undistorted image
    drawn_image = draw_detection(dst, persp_trans, Minv, ploty, left_fitx, right_fitx, radius_of_curvature, car_pos)
    if image_path is not None:
        save_image(cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR), image_path, 'drawn_image')

    return drawn_image

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
    Minv = cv2.getPerspectiveTransform(TRANSFORM_DEST_POINT, TRANSFORM_SRC_POINT)

    # # process test images
    # test_images = list(map(lambda s: 'test_images/' + s, os.listdir('test_images/')))
    # for image_path in test_images:
    #     # Read the image
    #     image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
    #     # Run the image through the pipeline
    #     detected_lane_image = detect_lane(image, image_path)
    #     left_fit = None
    #     right_fit = None
        
    #     # Save the image
    #     cv2.imwrite(image_path.replace('test_images','test_images_output').replace('.jpg', '')+'_detected.jpg', cv2.cvtColor(detected_lane_image, cv2.COLOR_RGB2BGR))

    # # process test video(s)
    # video = VideoFileClip("project_video.mp4")
    # #video.save_frame("frame.jpg", t=21.2) # saves the frame a t=2s
    # detected_lane_video = video.fl_image(detect_lane) #NOTE: this function expects color images!!
    # detected_lane_video.write_videofile("project_video_detected.mp4", audio=False)

    img_path = 'test_images/straight_lines1.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detect_lane(img, img_path)
    # img_path = 'test_images/test5.jpg'
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # detect_lane(img, img_path)
